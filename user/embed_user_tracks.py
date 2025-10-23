"""
Get or generate embeddings for user's listening history tracks.

Checks database first, generates new embeddings for missing tracks,
then projects all tracks to 2D using the pre-fitted UMAP model.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import joblib
import argparse

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from setup.database import EmbeddingDatabase
from setup.generate_embeddings import MERTEmbeddingGenerator

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserTrackEmbedder:
    """Manage embeddings for user tracks."""

    def __init__(self, device: str = None):
        """
        Initialize embedder.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.db = EmbeddingDatabase()
        self.embedding_generator = None  # Lazy initialization
        self.device = device
        self.umap_model = None

    def _init_embedding_generator(self):
        """Lazy initialization of MERT model (only when needed)."""
        if self.embedding_generator is None:
            logger.info("Initializing MERT embedding generator...")
            self.embedding_generator = MERTEmbeddingGenerator(device=self.device)

    def _load_umap_model(self):
        """Load the pre-fitted UMAP model."""
        if self.umap_model is None:
            if not config.UMAP_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"UMAP model not found at {config.UMAP_MODEL_PATH}. "
                    "Run train_umap.py first to create the base map."
                )

            logger.info(f"Loading UMAP model from {config.UMAP_MODEL_PATH}")
            self.umap_model = joblib.load(config.UMAP_MODEL_PATH)
            logger.info("UMAP model loaded successfully")

    def get_embeddings(self, track_ids: List[str], generate_missing: bool = True) -> Dict[str, np.ndarray]:
        """
        Get embeddings for a list of tracks.

        Args:
            track_ids: List of track IDs
            generate_missing: If True, generate embeddings for missing tracks

        Returns:
            Dict mapping track_id to 768-dim embedding
        """
        embeddings = {}
        missing_tracks = []

        # Check database for existing embeddings
        for track_id in track_ids:
            embedding = self.db.get_embedding(track_id)
            if embedding is not None:
                embeddings[track_id] = embedding
            else:
                missing_tracks.append(track_id)

        logger.info(f"Found {len(embeddings)}/{len(track_ids)} embeddings in database")

        # Generate missing embeddings if requested
        if missing_tracks and generate_missing:
            logger.info(f"Generating {len(missing_tracks)} missing embeddings...")
            self._init_embedding_generator()

            for track_id in missing_tracks:
                success = self.embedding_generator.process_track(track_id)
                if success:
                    # Retrieve the newly generated embedding
                    embedding = self.db.get_embedding(track_id)
                    if embedding is not None:
                        embeddings[track_id] = embedding

            logger.info(f"Generated {len(embeddings) - (len(track_ids) - len(missing_tracks))} new embeddings")

        # Save any new data
        if missing_tracks and generate_missing:
            self.db.save()

        return embeddings

    def project_to_2d(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """
        Project embeddings to 2D using pre-fitted UMAP model.

        CRITICAL: Uses transform() not fit_transform() to ensure
        new songs map to the same coordinate system.

        Args:
            embeddings: Dict mapping track_id to 768-dim embedding

        Returns:
            Dict mapping track_id to (x, y) coordinates
        """
        if not embeddings:
            return {}

        # Load UMAP model
        self._load_umap_model()

        # Prepare embeddings matrix
        track_ids = list(embeddings.keys())
        embeddings_matrix = np.array([embeddings[tid] for tid in track_ids])

        logger.info(f"Projecting {len(track_ids)} embeddings to 2D...")

        # Project using transform (not fit_transform!)
        coords = self.umap_model.transform(embeddings_matrix)

        # Create result dict
        coords_dict = {}
        for track_id, coord in zip(track_ids, coords):
            coords_dict[track_id] = (float(coord[0]), float(coord[1]))

        logger.info("Projection complete")
        return coords_dict

    def get_track_coords(self, track_ids: List[str], generate_missing: bool = True) -> Dict[str, Tuple[float, float]]:
        """
        Get 2D coordinates for tracks (end-to-end pipeline).

        Args:
            track_ids: List of track IDs
            generate_missing: If True, generate embeddings for missing tracks

        Returns:
            Dict mapping track_id to (x, y) coordinates
        """
        # Get embeddings
        embeddings = self.get_embeddings(track_ids, generate_missing=generate_missing)

        if not embeddings:
            logger.warning("No embeddings found for any tracks")
            return {}

        # Project to 2D
        coords = self.project_to_2d(embeddings)

        return coords

    def get_tracks_info(self, track_ids: List[str], generate_missing: bool = True) -> List[Dict]:
        """
        Get full information for tracks: metadata + coordinates.

        Args:
            track_ids: List of track IDs
            generate_missing: If True, generate embeddings for missing tracks

        Returns:
            List of dicts with track_id, metadata, and coordinates
        """
        # Get coordinates
        coords = self.get_track_coords(track_ids, generate_missing=generate_missing)

        # Get metadata
        tracks_info = []
        for track_id in track_ids:
            if track_id in coords:
                metadata = self.db.get_metadata(track_id)
                x, y = coords[track_id]

                track_info = {
                    'track_id': track_id,
                    'x': x,
                    'y': y
                }

                # Add metadata if available
                if metadata:
                    track_info.update({
                        'title': metadata.get('title', 'Unknown'),
                        'artist': metadata.get('artist', 'Unknown'),
                        'preview_url': metadata.get('preview_url'),
                        'popularity': metadata.get('popularity', 0)
                    })

                tracks_info.append(track_info)

        logger.info(f"Retrieved full info for {len(tracks_info)}/{len(track_ids)} tracks")
        return tracks_info


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Get or generate embeddings for user tracks")
    parser.add_argument(
        '--tracks',
        type=str,
        nargs='+',
        required=True,
        help='Track IDs to process'
    )
    parser.add_argument(
        '--no-generate',
        action='store_true',
        help='Do not generate missing embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use for embedding generation'
    )

    args = parser.parse_args()

    # Create embedder
    embedder = UserTrackEmbedder(device=args.device)

    # Process tracks
    generate_missing = not args.no_generate
    tracks_info = embedder.get_tracks_info(args.tracks, generate_missing=generate_missing)

    # Display results
    logger.info("=" * 60)
    logger.info(f"Track Information ({len(tracks_info)} tracks)")
    logger.info("=" * 60)

    for i, track in enumerate(tracks_info, 1):
        logger.info(f"\n{i}. {track.get('title', 'Unknown')} by {track.get('artist', 'Unknown')}")
        logger.info(f"   Track ID: {track['track_id']}")
        logger.info(f"   Coordinates: ({track['x']:.3f}, {track['y']:.3f})")
        logger.info(f"   Popularity: {track.get('popularity', 'N/A')}")


if __name__ == "__main__":
    main()
