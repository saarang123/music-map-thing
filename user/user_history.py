"""
Fetch user listening history from Spotify.

Retrieves recently played tracks for visualization on the music map.
"""

import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import argparse
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from setup.audio_source import SpotifySource

# Set up logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserHistoryFetcher:
    """Fetch and manage user listening history."""

    def __init__(self):
        """Initialize the fetcher."""
        self.spotify = SpotifySource()

    def get_history(self, timeframe: str = 'week') -> List[Tuple[str, datetime]]:
        """
        Get user's listening history for a specific timeframe.

        Args:
            timeframe: 'day', 'week', or 'month'

        Returns:
            List of (track_id, timestamp) tuples, sorted by timestamp (oldest first)
        """
        logger.info(f"Fetching listening history for timeframe: {timeframe}")

        history = self.spotify.get_user_listening_history(timeframe)

        if not history:
            logger.warning("No listening history found")
            return []

        # Sort by timestamp (oldest first) for chronological path visualization
        history.sort(key=lambda x: x[1])

        logger.info(f"Retrieved {len(history)} tracks from {timeframe}")
        return history

    def get_history_with_metadata(self, timeframe: str = 'week') -> List[dict]:
        """
        Get listening history with full track metadata.

        Returns:
            List of dicts with track_id, timestamp, and metadata
        """
        history = self.get_history(timeframe)

        if not history:
            return []

        # Get metadata for all tracks
        track_ids = [track_id for track_id, _ in history]
        metadata_list = self.spotify.get_tracks_metadata(track_ids)

        # Create mapping
        metadata_map = {m['track_id']: m for m in metadata_list}

        # Combine history with metadata
        result = []
        for track_id, timestamp in history:
            metadata = metadata_map.get(track_id)
            if metadata:
                result.append({
                    'track_id': track_id,
                    'timestamp': timestamp.isoformat(),
                    'title': metadata['title'],
                    'artist': metadata['artist'],
                    'preview_url': metadata['preview_url'],
                    'popularity': metadata['popularity']
                })
            else:
                logger.warning(f"Could not get metadata for track {track_id}")

        logger.info(f"Retrieved metadata for {len(result)}/{len(history)} tracks")
        return result

    def save_history(self, history: List[dict], output_path: Path):
        """Save history to JSON file."""
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump({
                'total_tracks': len(history),
                'tracks': history
            }, f, indent=2)

        logger.info(f"Saved history to {output_path}")

    def get_unique_tracks(self, history: List[Tuple[str, datetime]]) -> List[str]:
        """
        Get unique track IDs from history (for embedding generation).

        Args:
            history: List of (track_id, timestamp) tuples

        Returns:
            List of unique track IDs
        """
        unique_ids = list(set(track_id for track_id, _ in history))
        logger.info(f"Found {len(unique_ids)} unique tracks in history")
        return unique_ids

    def get_summary_stats(self, history: List[dict]) -> dict:
        """
        Get summary statistics about listening history.

        Args:
            history: List of track dicts with metadata

        Returns:
            Dict with summary statistics
        """
        if not history:
            return {
                'total_tracks': 0,
                'unique_tracks': 0,
                'unique_artists': 0,
                'date_range': None
            }

        # Extract data
        track_ids = [t['track_id'] for t in history]
        artists = [t['artist'] for t in history]
        timestamps = [datetime.fromisoformat(t['timestamp']) for t in history]

        stats = {
            'total_tracks': len(history),
            'unique_tracks': len(set(track_ids)),
            'unique_artists': len(set(artists)),
            'date_range': {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat()
            },
            'most_played_tracks': self._get_most_common(track_ids, history, 'track_id', 5),
            'most_played_artists': self._get_most_common(artists, history, 'artist', 5)
        }

        return stats

    def _get_most_common(self, items: List[str], history: List[dict],
                         key: str, limit: int) -> List[dict]:
        """Helper to get most common items."""
        from collections import Counter

        counts = Counter(items)
        most_common = counts.most_common(limit)

        result = []
        for item, count in most_common:
            # Find a representative track
            track = next(t for t in history if t[key] == item)
            result.append({
                key: item,
                'count': count,
                'title': track.get('title', ''),
                'artist': track.get('artist', '')
            })

        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch user listening history from Spotify")
    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['day', 'week', 'month'],
        default='week',
        help='Timeframe to fetch history for'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for history JSON'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show summary statistics'
    )

    args = parser.parse_args()

    # Create fetcher
    fetcher = UserHistoryFetcher()

    # Get history with metadata
    history = fetcher.get_history_with_metadata(timeframe=args.timeframe)

    if not history:
        logger.error("No listening history found")
        return

    # Print summary
    logger.info("=" * 60)
    logger.info(f"Listening History ({args.timeframe})")
    logger.info("=" * 60)
    logger.info(f"Total plays: {len(history)}")

    # Show first few tracks
    logger.info("\nRecent tracks:")
    for i, track in enumerate(history[:5], 1):
        logger.info(f"{i}. {track['title']} by {track['artist']}")
        logger.info(f"   Played at: {track['timestamp']}")

    # Show statistics if requested
    if args.stats:
        stats = fetcher.get_summary_stats(history)
        logger.info("\n" + "=" * 60)
        logger.info("Summary Statistics")
        logger.info("=" * 60)
        logger.info(f"Total plays: {stats['total_tracks']}")
        logger.info(f"Unique tracks: {stats['unique_tracks']}")
        logger.info(f"Unique artists: {stats['unique_artists']}")
        logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        logger.info("\nMost played tracks:")
        for i, track in enumerate(stats['most_played_tracks'], 1):
            logger.info(f"{i}. {track['title']} by {track['artist']} ({track['count']} plays)")

        logger.info("\nMost played artists:")
        for i, artist in enumerate(stats['most_played_artists'], 1):
            logger.info(f"{i}. {artist['artist']} ({artist['count']} plays)")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        fetcher.save_history(history, output_path)


if __name__ == "__main__":
    main()
