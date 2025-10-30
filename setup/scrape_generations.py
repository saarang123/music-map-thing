"""
Interface for collecting video generations for the base map.

NOTE: This is a simple interface. The actual data collection is handled
by your pipeline. This script just provides structure and examples.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import argparse

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_generations_from_json(json_path: Path) -> List[Dict]:
    """
    Load generations from JSON file.

    Expected format:
    [
        {
            "generation_id": "gen_001",
            "prompt": "a dragon flying over mountains",
            "image_url": "https://...",
            "category": "fantasy",
            "user_id": "user_123",  // optional
            "timestamp": "2024-10-23T14:30:00"  // optional
        },
        ...
    ]

    Args:
        json_path: Path to JSON file

    Returns:
        List of generation dicts
    """
    logger.info(f"Loading generations from {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both direct list and wrapped format
    if isinstance(data, dict) and 'generations' in data:
        generations = data['generations']
    elif isinstance(data, list):
        generations = data
    else:
        raise ValueError(f"Unexpected JSON format. Expected list or {{generations: [...]}}")

    logger.info(f"Loaded {len(generations)} generations")

    # Validate and normalize
    normalized = []
    for i, gen in enumerate(generations):
        if 'generation_id' not in gen and 'id' not in gen:
            logger.warning(f"Generation at index {i} missing ID, skipping")
            continue

        if 'prompt' not in gen:
            logger.warning(f"Generation {gen.get('generation_id', gen.get('id'))} missing prompt, skipping")
            continue

        if 'image_url' not in gen and 'image_path' not in gen:
            logger.warning(f"Generation {gen.get('generation_id', gen.get('id'))} missing image, skipping")
            continue

        # Normalize format
        normalized_gen = {
            'generation_id': gen.get('generation_id') or gen.get('id'),
            'prompt': gen['prompt'],
            'image_url': gen.get('image_url') or gen.get('image_path'),
            'category': gen.get('category', 'unknown'),
            'user_id': gen.get('user_id'),
            'timestamp': gen.get('timestamp') or gen.get('created_at')
        }
        normalized.append(normalized_gen)

    logger.info(f"Normalized {len(normalized)} valid generations")
    return normalized


def save_generations(generations: List[Dict], output_path: Path):
    """Save generations to JSON file."""
    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_data = {
        'total_generations': len(generations),
        'generations': generations,
        'categories': {}
    }

    # Count by category
    for gen in generations:
        cat = gen.get('category', 'unknown')
        output_data['categories'][cat] = output_data['categories'].get(cat, 0) + 1

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(generations)} generations to {output_path}")
    logger.info(f"Category breakdown: {output_data['categories']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load and normalize video generation data for embedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON format:
[
    {
        "generation_id": "gen_001",
        "prompt": "a dragon flying over mountains",
        "image_url": "https://example.com/image.jpg",
        "category": "fantasy"
    }
]

Or provide this data directly from your pipeline.
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with generation data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/base_generations.json)'
    )

    args = parser.parse_args()

    # Load generations
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    generations = load_generations_from_json(input_path)

    if not generations:
        logger.error("No valid generations found")
        return 1

    # Save normalized output
    output_path = Path(args.output) if args.output else config.BASE_GENERATIONS_PATH
    save_generations(generations, output_path)

    logger.info("=" * 60)
    logger.info("Generation data ready")
    logger.info("=" * 60)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Generate embeddings: python setup/embed_generations.py")
    logger.info(f"2. Train UMAP: python setup/train_umap.py")

    return 0


if __name__ == "__main__":
    exit(main())
