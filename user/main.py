"""
Main entry point for generating user music map visualizations.

End-to-end pipeline:
1. Fetch user listening history
2. Get/generate embeddings for tracks
3. Project to 2D coordinates
4. Create visualization
"""

import logging
from pathlib import Path
import argparse
from datetime import datetime
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from user.user_history import UserHistoryFetcher
from user.embed_user_tracks import UserTrackEmbedder
from user.visualize import MusicMapVisualizer

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


class MusicMapPipeline:
    """End-to-end pipeline for generating user music maps."""

    def __init__(self, device: str = None):
        """
        Initialize pipeline.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.history_fetcher = UserHistoryFetcher()
        self.track_embedder = UserTrackEmbedder(device=device)
        self.visualizer = MusicMapVisualizer()

    def generate_music_map(
        self,
        timeframe: str = 'week',
        output_dir: Path = None,
        viz_format: str = 'both',
        show_stats: bool = True
    ) -> dict:
        """
        Generate complete music map for user.

        Args:
            timeframe: 'day', 'week', or 'month'
            output_dir: Where to save outputs (default: data/visualizations/)
            viz_format: 'matplotlib', 'plotly', 'both', or 'heatmap'
            show_stats: Whether to display statistics

        Returns:
            Dict with paths to generated files
        """
        logger.info("=" * 60)
        logger.info("Music Map Generation Pipeline")
        logger.info("=" * 60)
        logger.info(f"Timeframe: {timeframe}")

        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.VIZ_OUTPUT_DIR / f"session_{timestamp}"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Step 1: Fetch listening history
        logger.info("\n[1/4] Fetching listening history...")
        history = self.history_fetcher.get_history_with_metadata(timeframe=timeframe)

        if not history:
            logger.error("No listening history found")
            return {}

        logger.info(f"Found {len(history)} tracks")

        # Show statistics if requested
        if show_stats:
            stats = self.history_fetcher.get_summary_stats(history)
            logger.info("\nListening Statistics:")
            logger.info(f"  Total plays: {stats['total_tracks']}")
            logger.info(f"  Unique tracks: {stats['unique_tracks']}")
            logger.info(f"  Unique artists: {stats['unique_artists']}")

            if stats['most_played_tracks']:
                logger.info("\n  Top 3 tracks:")
                for i, track in enumerate(stats['most_played_tracks'][:3], 1):
                    logger.info(f"    {i}. {track['title']} by {track['artist']} ({track['count']} plays)")

        # Save history
        history_file = output_dir / "listening_history.json"
        self.history_fetcher.save_history(history, history_file)
        logger.info(f"Saved history to {history_file}")

        # Step 2: Get track IDs
        logger.info("\n[2/4] Processing tracks...")
        track_ids = [t['track_id'] for t in history]

        # Step 3: Get embeddings and coordinates
        logger.info("\n[3/4] Getting embeddings and projecting to 2D...")
        tracks_info = self.track_embedder.get_tracks_info(track_ids, generate_missing=True)

        if not tracks_info:
            logger.error("Could not get coordinates for any tracks")
            return {}

        logger.info(f"Successfully processed {len(tracks_info)}/{len(track_ids)} tracks")

        # Combine with timestamps
        timestamp_map = {t['track_id']: t['timestamp'] for t in history}
        for track in tracks_info:
            track['timestamp'] = timestamp_map.get(track['track_id'], '')

        # Sort by timestamp
        tracks_info.sort(key=lambda t: t.get('timestamp', ''))

        # Save tracks with coordinates
        tracks_file = output_dir / "tracks_with_coords.json"
        with open(tracks_file, 'w') as f:
            json.dump({'tracks': tracks_info}, f, indent=2)
        logger.info(f"Saved track coordinates to {tracks_file}")

        # Step 4: Create visualizations
        logger.info("\n[4/4] Creating visualizations...")
        output_files = {
            'history': str(history_file),
            'tracks': str(tracks_file)
        }

        title = f"My Music Journey ({timeframe})"

        if viz_format in ['matplotlib', 'both']:
            png_file = output_dir / "music_map.png"
            self.visualizer.create_matplotlib_visualization(
                tracks_info,
                png_file,
                title=title
            )
            output_files['png'] = str(png_file)

        if viz_format in ['plotly', 'both']:
            html_file = output_dir / "music_map.html"
            self.visualizer.create_plotly_visualization(
                tracks_info,
                html_file,
                title=title
            )
            output_files['html'] = str(html_file)

        if viz_format == 'heatmap':
            heatmap_file = output_dir / "music_heatmap.png"
            self.visualizer.create_heatmap_visualization(
                tracks_info,
                heatmap_file,
                title=f"Listening Heat Map ({timeframe})"
            )
            output_files['heatmap'] = str(heatmap_file)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Music Map Generation Complete!")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info("\nGenerated files:")
        for key, path in output_files.items():
            logger.info(f"  {key}: {path}")
        logger.info("=" * 60)

        return output_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate music map visualization from Spotify listening history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate map for last week (default)
  python main.py

  # Generate map for last month
  python main.py --timeframe month

  # Generate only interactive HTML visualization
  python main.py --format plotly

  # Specify custom output directory
  python main.py --output ./my_music_maps
        """
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['day', 'week', 'month'],
        default='week',
        help='Timeframe for listening history (default: week)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: auto-generated in data/visualizations/)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['matplotlib', 'plotly', 'both', 'heatmap'],
        default='both',
        help='Visualization format (default: both)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device for embedding generation (default: auto-detect)'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Do not show listening statistics'
    )

    args = parser.parse_args()

    try:
        # Create pipeline
        pipeline = MusicMapPipeline(device=args.device)

        # Generate music map
        output_files = pipeline.generate_music_map(
            timeframe=args.timeframe,
            output_dir=Path(args.output) if args.output else None,
            viz_format=args.format,
            show_stats=not args.no_stats
        )

        if not output_files:
            logger.error("Failed to generate music map")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error generating music map: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
