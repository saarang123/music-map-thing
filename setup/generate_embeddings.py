"""
Generate MERT embeddings for songs.

Processes audio files through the MERT model to extract 768-dimensional embeddings.
Includes checkpointing for fault tolerance and progress tracking.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import tempfile
import os

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from setup.audio_source import SpotifySource, download_audio
from setup.database import EmbeddingDatabase

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MERTEmbeddingGenerator:
    """Generate MERT embeddings for audio tracks."""

    def __init__(self, device: str = None):
        """
        Initialize MERT model and processor.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load MERT model and processor
        logger.info(f"Loading MERT model: {config.MERT_MODEL_NAME}")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(config.MERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(config.MERT_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("MERT model loaded successfully")

        # Initialize components
        self.spotify = SpotifySource()
        self.db = EmbeddingDatabase()

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_no_preview': 0,
            'skipped_already_exists': 0
        }

    def load_and_preprocess_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess audio file for MERT.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio tensor or None if failed
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 24kHz if needed (MERT expects this)
            if sample_rate != config.MERT_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=config.MERT_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Process through MERT processor
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=config.MERT_SAMPLE_RATE,
                return_tensors="pt"
            )

            return inputs

        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_path}: {e}")
            return None

    def generate_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Generate MERT embedding for an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            768-dimensional embedding or None if failed
        """
        try:
            # Load and preprocess audio
            inputs = self.load_and_preprocess_audio(audio_path)
            if inputs is None:
                return None

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract embedding (mean pooling over time dimension)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            # Move to CPU and convert to numpy
            embedding = embedding.cpu().numpy()

            # Verify shape
            if embedding.shape != (config.EMBEDDING_DIM,):
                logger.error(f"Unexpected embedding shape: {embedding.shape}")
                return None

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for {audio_path}: {e}")
            return None

    def process_track(self, track_id: str, metadata: Dict = None, force: bool = False) -> bool:
        """
        Process a single track: download audio, generate embedding, store.

        Args:
            track_id: Spotify track ID
            metadata: Track metadata (if already fetched)
            force: Force reprocessing even if embedding exists

        Returns:
            True if successful, False otherwise
        """
        # Check if already processed
        if not force and self.db.has_embedding(track_id):
            logger.debug(f"Skipping {track_id}: already processed")
            self.stats['skipped_already_exists'] += 1
            return True

        # Get metadata if not provided
        if metadata is None:
            metadata = self.spotify.get_track_metadata(track_id)
            if metadata is None:
                logger.warning(f"Could not fetch metadata for {track_id}")
                self.stats['failed'] += 1
                return False

        # Check if preview URL exists
        preview_url = metadata.get('preview_url')
        if not preview_url:
            logger.warning(f"No preview URL for {track_id} ({metadata.get('title', 'Unknown')})")
            self.stats['skipped_no_preview'] += 1
            return False

        # Download audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Download audio
            if not download_audio(preview_url, tmp_path):
                logger.error(f"Failed to download audio for {track_id}")
                self.stats['failed'] += 1
                return False

            # Generate embedding
            embedding = self.generate_embedding(tmp_path)
            if embedding is None:
                logger.error(f"Failed to generate embedding for {track_id}")
                self.stats['failed'] += 1
                return False

            # Store in database
            self.db.add_embedding(
                track_id=track_id,
                embedding=embedding,
                metadata=metadata
            )

            logger.debug(f"Successfully processed {track_id}: {metadata.get('title')} by {metadata.get('artist')}")
            self.stats['successful'] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing track {track_id}: {e}")
            self.stats['failed'] += 1
            return False

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self.stats['processed'] += 1

    def process_batch(self, track_data: List[Dict], force: bool = False):
        """
        Process a batch of tracks.

        Args:
            track_data: List of dicts with 'track_id' and optional metadata
            force: Force reprocessing even if embeddings exist
        """
        for track in tqdm(track_data, desc="Processing tracks"):
            track_id = track.get('track_id') or track.get('id')
            metadata = track if 'title' in track else None

            self.process_track(track_id, metadata=metadata, force=force)

            # Save checkpoint periodically
            if self.stats['processed'] % config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        """Save current progress."""
        self.db.save()
        logger.info(f"Checkpoint saved - Stats: {self.stats}")

    def process_from_json(self, json_path: Path, force: bool = False):
        """
        Process tracks from scraped JSON file.

        Args:
            json_path: Path to JSON file with track data
            force: Force reprocessing
        """
        logger.info(f"Loading tracks from {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        tracks = data.get('tracks', {})
        track_list = []

        for track_id, metadata in tracks.items():
            track_list.append({
                'track_id': track_id,
                **metadata
            })

        logger.info(f"Loaded {len(track_list)} tracks")

        # Process all tracks
        self.process_batch(track_list, force=force)

        # Final save
        self.save_checkpoint()

        # Print final statistics
        logger.info("=" * 60)
        logger.info("Embedding generation complete")
        logger.info(f"Total processed: {self.stats['processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped (no preview): {self.stats['skipped_no_preview']}")
        logger.info(f"Skipped (already exists): {self.stats['skipped_already_exists']}")
        logger.info("=" * 60)

        # Database stats
        db_stats = self.db.get_stats()
        logger.info(f"Database now contains {db_stats['total_tracks']} track embeddings")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate MERT embeddings for songs")
    parser.add_argument(
        '--input',
        type=str,
        default=str(config.BASE_SONGS_PATH),
        help='Input JSON file with track data'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of existing embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use (cpu or cuda). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--single-track',
        type=str,
        default=None,
        help='Process a single track by ID (for testing)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = MERTEmbeddingGenerator(device=args.device)

    if args.single_track:
        # Test mode: process single track
        logger.info(f"Processing single track: {args.single_track}")
        success = generator.process_track(args.single_track, force=args.force)
        if success:
            logger.info("Successfully processed track")
        else:
            logger.error("Failed to process track")
    else:
        # Process from JSON file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        generator.process_from_json(input_path, force=args.force)


if __name__ == "__main__":
    main()
