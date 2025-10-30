"""
Generate CLIP embeddings for video generations (image + prompt).

Uses CLIP to create combined embeddings that capture both visual and semantic information.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
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


class CLIPEmbeddingGenerator:
    """Generate CLIP embeddings for video generations (image + text)."""

    def __init__(self, device: str = None):
        """
        Initialize CLIP model and processor.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load CLIP model and processor
        logger.info(f"Loading CLIP model: {config.CLIP_MODEL_NAME}")
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully")

        # Initialize database
        self.db = EmbeddingDatabase()

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_already_exists': 0,
            'skipped_no_image': 0
        }

    def load_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """
        Load image from URL or local path.

        Args:
            image_url: URL or local file path

        Returns:
            PIL Image or None if failed
        """
        try:
            # Check if it's a URL
            if image_url.startswith('http://') or image_url.startswith('https://'):
                response = requests.get(image_url, timeout=config.TIMEOUT)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # Local file path
                image = Image.open(image_url)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large (for faster processing)
            if config.IMAGE_MAX_SIZE:
                image.thumbnail(config.IMAGE_MAX_SIZE, Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            logger.error(f"Failed to load image from {image_url}: {e}")
            return None

    def generate_embedding(
        self,
        image: Image.Image,
        prompt: str
    ) -> Optional[np.ndarray]:
        """
        Generate combined CLIP embedding for image + text.

        Args:
            image: PIL Image
            prompt: Text prompt

        Returns:
            Combined embedding (768-dim or 1536-dim depending on config)
        """
        try:
            # Process image and text
            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embed = outputs.image_embeds[0]  # 768-dim
                text_embed = outputs.text_embeds[0]    # 768-dim

            # Combine embeddings based on config
            if config.COMBINE_METHOD == "average":
                # Average: maintains 768-dim
                combined = (image_embed + text_embed) / 2
            elif config.COMBINE_METHOD == "concat":
                # Concatenate: creates 1536-dim
                combined = torch.cat([image_embed, text_embed])
            else:
                raise ValueError(f"Unknown combine method: {config.COMBINE_METHOD}")

            # Move to CPU and convert to numpy
            embedding = combined.cpu().numpy()

            # Verify shape
            expected_dim = config.EMBEDDING_DIM
            if embedding.shape != (expected_dim,):
                logger.error(f"Unexpected embedding shape: {embedding.shape}, expected ({expected_dim},)")
                return None

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def process_generation(
        self,
        generation: Dict,
        force: bool = False
    ) -> bool:
        """
        Process a single generation: load image, generate embedding, store.

        Args:
            generation: Dict with keys: generation_id, prompt, image_url, category, etc.
            force: Force reprocessing even if embedding exists

        Returns:
            True if successful, False otherwise
        """
        generation_id = generation.get('generation_id') or generation.get('id')
        prompt = generation.get('prompt', '')
        image_url = generation.get('image_url') or generation.get('image_path')
        category = generation.get('category', 'unknown')

        if not generation_id:
            logger.warning("Generation missing ID, skipping")
            self.stats['failed'] += 1
            return False

        # Check if already processed
        if not force and self.db.has_embedding(generation_id):
            logger.debug(f"Skipping {generation_id}: already processed")
            self.stats['skipped_already_exists'] += 1
            return True

        # Check if image URL exists
        if not image_url:
            logger.warning(f"No image URL for generation {generation_id}")
            self.stats['skipped_no_image'] += 1
            return False

        try:
            # Load image
            image = self.load_image_from_url(image_url)
            if image is None:
                logger.error(f"Failed to load image for {generation_id}")
                self.stats['failed'] += 1
                return False

            # Generate embedding
            embedding = self.generate_embedding(image, prompt)
            if embedding is None:
                logger.error(f"Failed to generate embedding for {generation_id}")
                self.stats['failed'] += 1
                return False

            # Prepare metadata
            metadata = {
                'generation_id': generation_id,
                'prompt': prompt,
                'image_url': image_url,
                'category': category,
                'user_id': generation.get('user_id'),
                'timestamp': generation.get('timestamp')
            }

            # Store in database
            self.db.add_embedding(
                track_id=generation_id,  # Using track_id for compatibility with existing DB
                embedding=embedding,
                metadata=metadata
            )

            logger.debug(f"Successfully processed {generation_id}: {prompt[:50]}...")
            self.stats['successful'] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing generation {generation_id}: {e}")
            self.stats['failed'] += 1
            return False

        finally:
            self.stats['processed'] += 1

    def process_batch(
        self,
        generations: List[Dict],
        force: bool = False
    ):
        """
        Process a batch of generations.

        Args:
            generations: List of generation dicts
            force: Force reprocessing
        """
        for generation in tqdm(generations, desc="Processing generations"):
            self.process_generation(generation, force=force)

            # Save checkpoint periodically
            if self.stats['processed'] % config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        """Save current progress."""
        self.db.save()
        logger.info(f"Checkpoint saved - Stats: {self.stats}")

    def process_from_list(
        self,
        generations: List[Dict],
        force: bool = False
    ):
        """
        Process generations from a list.

        Args:
            generations: List of generation dicts
            force: Force reprocessing
        """
        logger.info(f"Processing {len(generations)} generations")

        # Process all generations
        self.process_batch(generations, force=force)

        # Final save
        self.save_checkpoint()

        # Print final statistics
        logger.info("=" * 60)
        logger.info("Embedding generation complete")
        logger.info(f"Total processed: {self.stats['processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped (no image): {self.stats['skipped_no_image']}")
        logger.info(f"Skipped (already exists): {self.stats['skipped_already_exists']}")
        logger.info("=" * 60)

        # Database stats
        db_stats = self.db.get_stats()
        logger.info(f"Database now contains {db_stats['total_tracks']} generation embeddings")


def main():
    """Main entry point - expects you to provide generation data."""
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for video generations"
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use (cpu or cuda). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of existing embeddings'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with sample data'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = CLIPEmbeddingGenerator(device=args.device)

    if args.test:
        # Test mode: create sample generation
        logger.info("Running in test mode with sample data")
        sample_generations = [
            {
                'generation_id': 'test_001',
                'prompt': 'a dragon flying over mountains at sunset',
                'image_url': 'https://picsum.photos/512/512',  # Random image for testing
                'category': 'fantasy'
            }
        ]
        generator.process_from_list(sample_generations, force=args.force)
    else:
        logger.info("=" * 60)
        logger.info("Ready to process generations")
        logger.info("=" * 60)
        logger.info("\nUsage: Call this module from your data pipeline:")
        logger.info("\n  from setup.embed_generations import CLIPEmbeddingGenerator")
        logger.info("  generator = CLIPEmbeddingGenerator()")
        logger.info("  generator.process_from_list(your_generations_list)")
        logger.info("\nOr use --test flag to test with sample data")


if __name__ == "__main__":
    main()
