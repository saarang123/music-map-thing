"""
Collect 100K diverse songs from Spotify to build the base music universe.

Distribution:
- 10K: Global top songs
- 70K: Genre diversity (~500-600 songs from each of ~125 genres)
- 10K: Geographic diversity
- 10K: Recent releases

Output: JSON file with list of track IDs and metadata
"""

import json
import logging
from pathlib import Path
from typing import List, Set
from tqdm import tqdm
import time
import argparse

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


class SongScraper:
    """Collect diverse songs from Spotify."""

    def __init__(self):
        """Initialize the scraper."""
        self.spotify = SpotifySource()
        self.collected_tracks = {}  # track_id -> metadata
        self.track_ids_set = set()  # For deduplication

    def add_track(self, track_id: str, metadata: dict = None):
        """Add a track if not already collected."""
        if track_id not in self.track_ids_set:
            self.track_ids_set.add(track_id)
            self.collected_tracks[track_id] = metadata or {}
            return True
        return False

    def collect_global_top_tracks(self, target: int = config.TARGET_GLOBAL_TOP) -> int:
        """
        Collect global top tracks.
        Uses Spotify's Top 50 playlists and searches for popular tracks.
        """
        logger.info(f"Collecting {target} global top tracks...")
        initial_count = len(self.track_ids_set)

        try:
            # Get Global Top 50
            top_tracks = self.spotify.get_global_top_tracks(limit=50)
            for track_id in top_tracks:
                metadata = self.spotify.get_track_metadata(track_id)
                if metadata:
                    self.add_track(track_id, {**metadata, 'source': 'global_top'})

            # Get more popular tracks through searches
            popular_searches = [
                "top hits", "pop hits", "viral hits", "trending",
                "billboard", "charts", "most popular", "hot 100"
            ]

            with tqdm(total=target, desc="Global top tracks") as pbar:
                pbar.update(len(self.track_ids_set) - initial_count)

                for query in popular_searches:
                    if len(self.track_ids_set) - initial_count >= target:
                        break

                    results = self.spotify.search_tracks(query, limit=50)
                    for track in results:
                        if len(self.track_ids_set) - initial_count >= target:
                            break

                        self.add_track(
                            track['track_id'],
                            {**track, 'source': 'global_top'}
                        )
                        pbar.update(1)

                    time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.error(f"Error collecting global top tracks: {e}")

        collected = len(self.track_ids_set) - initial_count
        logger.info(f"Collected {collected} global top tracks")
        return collected

    def collect_genre_diversity(self, target: int = config.TARGET_GENRE_DIVERSITY) -> int:
        """
        Collect tracks across diverse genres.
        ~500-600 songs per genre from 125 genre seeds.
        """
        logger.info(f"Collecting {target} tracks for genre diversity...")
        initial_count = len(self.track_ids_set)

        genres = config.SPOTIFY_GENRES
        tracks_per_genre = target // len(genres)

        logger.info(f"Targeting ~{tracks_per_genre} tracks per genre across {len(genres)} genres")

        with tqdm(total=len(genres), desc="Genres processed") as pbar:
            for genre in genres:
                genre_tracks_added = 0

                try:
                    # Get recommendations for this genre
                    track_ids = self.spotify.get_tracks_by_genre(genre, limit=50)

                    # Get metadata for each track
                    for track_id in track_ids:
                        if genre_tracks_added >= tracks_per_genre:
                            break

                        metadata = self.spotify.get_track_metadata(track_id)
                        if metadata:
                            added = self.add_track(
                                track_id,
                                {**metadata, 'source': 'genre_diversity', 'genre': genre}
                            )
                            if added:
                                genre_tracks_added += 1

                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.warning(f"Error collecting tracks for genre '{genre}': {e}")

                pbar.update(1)

        collected = len(self.track_ids_set) - initial_count
        logger.info(f"Collected {collected} tracks for genre diversity")
        return collected

    def collect_geographic_diversity(self, target: int = config.TARGET_GEOGRAPHIC_DIVERSITY) -> int:
        """
        Collect tracks from different geographic markets.
        """
        logger.info(f"Collecting {target} tracks for geographic diversity...")
        initial_count = len(self.track_ids_set)

        markets = config.GEOGRAPHIC_MARKETS
        tracks_per_market = target // len(markets)

        logger.info(f"Targeting ~{tracks_per_market} tracks per market across {len(markets)} markets")

        with tqdm(total=len(markets), desc="Markets processed") as pbar:
            for market in markets:
                market_tracks_added = 0

                try:
                    track_ids = self.spotify.get_tracks_by_market(market, limit=50)

                    for track_id in track_ids:
                        if market_tracks_added >= tracks_per_market:
                            break

                        metadata = self.spotify.get_track_metadata(track_id)
                        if metadata:
                            added = self.add_track(
                                track_id,
                                {**metadata, 'source': 'geographic_diversity', 'market': market}
                            )
                            if added:
                                market_tracks_added += 1

                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.warning(f"Error collecting tracks for market '{market}': {e}")

                pbar.update(1)

        collected = len(self.track_ids_set) - initial_count
        logger.info(f"Collected {collected} tracks for geographic diversity")
        return collected

    def collect_recent_releases(self, target: int = config.TARGET_RECENT_RELEASES) -> int:
        """
        Collect recent releases (last 6 months).
        """
        logger.info(f"Collecting {target} recent release tracks...")
        initial_count = len(self.track_ids_set)

        try:
            # Search for recent releases
            recent_queries = [
                "new music 2024", "new releases", "latest songs",
                "new singles", "fresh finds", "recently released"
            ]

            with tqdm(total=target, desc="Recent releases") as pbar:
                for query in recent_queries:
                    if len(self.track_ids_set) - initial_count >= target:
                        break

                    results = self.spotify.search_tracks(query, limit=50)
                    for track in results:
                        if len(self.track_ids_set) - initial_count >= target:
                            break

                        self.add_track(
                            track['track_id'],
                            {**track, 'source': 'recent_releases'}
                        )
                        pbar.update(1)

                    time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.error(f"Error collecting recent releases: {e}")

        collected = len(self.track_ids_set) - initial_count
        logger.info(f"Collected {collected} recent release tracks")
        return collected

    def save_results(self, output_path: Path = config.BASE_SONGS_PATH):
        """Save collected tracks to JSON file."""
        output_path.parent.mkdir(exist_ok=True, parents=True)

        output_data = {
            'total_tracks': len(self.collected_tracks),
            'tracks': self.collected_tracks,
            'statistics': {
                'global_top': sum(1 for t in self.collected_tracks.values() if t.get('source') == 'global_top'),
                'genre_diversity': sum(1 for t in self.collected_tracks.values() if t.get('source') == 'genre_diversity'),
                'geographic_diversity': sum(1 for t in self.collected_tracks.values() if t.get('source') == 'geographic_diversity'),
                'recent_releases': sum(1 for t in self.collected_tracks.values() if t.get('source') == 'recent_releases'),
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved {len(self.collected_tracks)} tracks to {output_path}")
        logger.info(f"Statistics: {output_data['statistics']}")

    def run(self, subset_size: int = None):
        """
        Run the full collection process.

        Args:
            subset_size: If provided, collect only this many songs total (for testing)
        """
        logger.info("=" * 60)
        logger.info("Starting song collection")
        logger.info("=" * 60)

        if subset_size:
            logger.info(f"Running in SUBSET mode: collecting {subset_size} songs")
            # Adjust targets proportionally
            total_target = subset_size
            global_target = int(config.TARGET_GLOBAL_TOP * subset_size / config.TARGET_TOTAL_SONGS)
            genre_target = int(config.TARGET_GENRE_DIVERSITY * subset_size / config.TARGET_TOTAL_SONGS)
            geo_target = int(config.TARGET_GEOGRAPHIC_DIVERSITY * subset_size / config.TARGET_TOTAL_SONGS)
            recent_target = int(config.TARGET_RECENT_RELEASES * subset_size / config.TARGET_TOTAL_SONGS)
        else:
            total_target = config.TARGET_TOTAL_SONGS
            global_target = config.TARGET_GLOBAL_TOP
            genre_target = config.TARGET_GENRE_DIVERSITY
            geo_target = config.TARGET_GEOGRAPHIC_DIVERSITY
            recent_target = config.TARGET_RECENT_RELEASES

        # Collect from each source
        self.collect_global_top_tracks(global_target)
        self.collect_genre_diversity(genre_target)
        self.collect_geographic_diversity(geo_target)
        self.collect_recent_releases(recent_target)

        # Save results
        self.save_results()

        logger.info("=" * 60)
        logger.info(f"Collection complete: {len(self.collected_tracks)} unique tracks")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect diverse songs from Spotify")
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Collect only a subset of songs (for testing). E.g., --subset 1000'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/base_songs.json)'
    )

    args = parser.parse_args()

    # Override output path if specified
    if args.output:
        config.BASE_SONGS_PATH = Path(args.output)

    scraper = SongScraper()
    scraper.run(subset_size=args.subset)


if __name__ == "__main__":
    main()
