"""
Abstract interface for audio sources with Spotify implementation.
Designed to be modular - easy to swap to Apple Music or other services later.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """
    Abstract base class for audio sources.
    Implement this interface to add support for different music services.
    """

    @abstractmethod
    def get_track_metadata(self, track_id: str) -> Optional[Dict]:
        """
        Get metadata for a single track.

        Returns:
            Dict with keys: track_id, title, artist, preview_url, duration_ms, popularity
            None if track not found
        """
        pass

    @abstractmethod
    def get_tracks_metadata(self, track_ids: List[str]) -> List[Dict]:
        """Get metadata for multiple tracks (batch operation)."""
        pass

    @abstractmethod
    def get_preview_url(self, track_id: str) -> Optional[str]:
        """Get preview URL for a track."""
        pass

    @abstractmethod
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for tracks by query string."""
        pass

    @abstractmethod
    def get_user_listening_history(self, timeframe: str = 'week') -> List[Tuple[str, datetime]]:
        """
        Get user's listening history.

        Args:
            timeframe: 'day', 'week', or 'month'

        Returns:
            List of (track_id, timestamp) tuples
        """
        pass

    @abstractmethod
    def get_global_top_tracks(self, limit: int = 50) -> List[str]:
        """Get global top tracks."""
        pass

    @abstractmethod
    def get_tracks_by_genre(self, genre: str, limit: int = 50) -> List[str]:
        """Get tracks for a specific genre."""
        pass

    @abstractmethod
    def get_tracks_by_market(self, market: str, limit: int = 50) -> List[str]:
        """Get top tracks for a specific geographic market."""
        pass


class SpotifySource(AudioSource):
    """
    Spotify implementation of AudioSource interface.
    """

    def __init__(self, client_id: str = None, client_secret: str = None, redirect_uri: str = None):
        """
        Initialize Spotify client.

        Args:
            client_id: Spotify client ID (defaults to config)
            client_secret: Spotify client secret (defaults to config)
            redirect_uri: Redirect URI for OAuth (defaults to config)
        """
        self.client_id = client_id or config.SPOTIFY_CLIENT_ID
        self.client_secret = client_secret or config.SPOTIFY_CLIENT_SECRET
        self.redirect_uri = redirect_uri or config.SPOTIFY_REDIRECT_URI

        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

        # Client credentials flow (for non-user data)
        auth_manager = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

        # User auth client (initialized only when needed)
        self.sp_user = None

        logger.info("SpotifySource initialized successfully")

    def _init_user_auth(self):
        """Initialize user authentication (OAuth) for accessing user data."""
        if self.sp_user is None:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope="user-read-recently-played user-top-read"
            )
            self.sp_user = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("User authentication initialized")

    def _retry_api_call(self, func, *args, **kwargs):
        """Retry API calls with exponential backoff."""
        for attempt in range(config.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    logger.error(f"API call failed after {config.MAX_RETRIES} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt + 1}/{config.MAX_RETRIES}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

    def get_track_metadata(self, track_id: str) -> Optional[Dict]:
        """Get metadata for a single track."""
        try:
            track = self._retry_api_call(self.sp.track, track_id)

            return {
                'track_id': track['id'],
                'title': track['name'],
                'artist': ', '.join([artist['name'] for artist in track['artists']]),
                'preview_url': track.get('preview_url'),
                'duration_ms': track['duration_ms'],
                'popularity': track.get('popularity', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for track {track_id}: {e}")
            return None

    def get_tracks_metadata(self, track_ids: List[str]) -> List[Dict]:
        """Get metadata for multiple tracks (batch operation)."""
        results = []

        # Spotify allows max 50 tracks per request
        batch_size = 50
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            try:
                tracks = self._retry_api_call(self.sp.tracks, batch)

                for track in tracks['tracks']:
                    if track is None:
                        continue

                    results.append({
                        'track_id': track['id'],
                        'title': track['name'],
                        'artist': ', '.join([artist['name'] for artist in track['artists']]),
                        'preview_url': track.get('preview_url'),
                        'duration_ms': track['duration_ms'],
                        'popularity': track.get('popularity', 0)
                    })
            except Exception as e:
                logger.error(f"Failed to get metadata for batch: {e}")

        return results

    def get_preview_url(self, track_id: str) -> Optional[str]:
        """Get preview URL for a track."""
        metadata = self.get_track_metadata(track_id)
        return metadata['preview_url'] if metadata else None

    def search_tracks(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for tracks by query string."""
        try:
            results = self._retry_api_call(self.sp.search, q=query, type='track', limit=limit)

            tracks = []
            for item in results['tracks']['items']:
                tracks.append({
                    'track_id': item['id'],
                    'title': item['name'],
                    'artist': ', '.join([artist['name'] for artist in item['artists']]),
                    'preview_url': item.get('preview_url'),
                    'duration_ms': item['duration_ms'],
                    'popularity': item.get('popularity', 0)
                })

            return tracks
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def get_user_listening_history(self, timeframe: str = 'week') -> List[Tuple[str, datetime]]:
        """Get user's listening history."""
        self._init_user_auth()

        # Calculate time threshold
        now = datetime.now()
        if timeframe == 'day':
            after = int((now - timedelta(days=1)).timestamp() * 1000)
        elif timeframe == 'week':
            after = int((now - timedelta(weeks=1)).timestamp() * 1000)
        elif timeframe == 'month':
            after = int((now - timedelta(days=30)).timestamp() * 1000)
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        history = []
        try:
            # Fetch recently played tracks
            results = self._retry_api_call(self.sp_user.current_user_recently_played, limit=50, after=after)

            for item in results['items']:
                track_id = item['track']['id']
                played_at = datetime.strptime(item['played_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
                history.append((track_id, played_at))

            logger.info(f"Retrieved {len(history)} tracks from {timeframe} listening history")
            return history

        except Exception as e:
            logger.error(f"Failed to get listening history: {e}")
            return []

    def get_global_top_tracks(self, limit: int = 50) -> List[str]:
        """
        Get global top tracks.
        Uses the Global Top 50 playlist as a proxy.
        """
        try:
            # Spotify's Global Top 50 playlist ID
            playlist_id = "37i9dQZEVXbMDoHDwVN2tF"

            results = self._retry_api_call(self.sp.playlist_tracks, playlist_id, limit=limit)

            track_ids = []
            for item in results['items']:
                if item['track']:
                    track_ids.append(item['track']['id'])

            logger.info(f"Retrieved {len(track_ids)} global top tracks")
            return track_ids

        except Exception as e:
            logger.error(f"Failed to get global top tracks: {e}")
            return []

    def get_tracks_by_genre(self, genre: str, limit: int = 50) -> List[str]:
        """Get tracks for a specific genre using recommendations."""
        try:
            # Use Spotify's recommendations endpoint with genre seed
            results = self._retry_api_call(
                self.sp.recommendations,
                seed_genres=[genre],
                limit=limit
            )

            track_ids = [track['id'] for track in results['tracks']]
            logger.info(f"Retrieved {len(track_ids)} tracks for genre '{genre}'")
            return track_ids

        except Exception as e:
            logger.error(f"Failed to get tracks for genre '{genre}': {e}")
            return []

    def get_tracks_by_market(self, market: str, limit: int = 50) -> List[str]:
        """
        Get top tracks for a specific geographic market.
        Uses featured playlists for the market.
        """
        try:
            # Get featured playlists for the market
            playlists = self._retry_api_call(
                self.sp.featured_playlists,
                country=market,
                limit=1
            )

            if not playlists['playlists']['items']:
                logger.warning(f"No playlists found for market {market}")
                return []

            # Get tracks from the first featured playlist
            playlist_id = playlists['playlists']['items'][0]['id']
            results = self._retry_api_call(self.sp.playlist_tracks, playlist_id, limit=limit)

            track_ids = []
            for item in results['items']:
                if item['track']:
                    track_ids.append(item['track']['id'])

            logger.info(f"Retrieved {len(track_ids)} tracks for market {market}")
            return track_ids

        except Exception as e:
            logger.error(f"Failed to get tracks for market {market}: {e}")
            return []


def download_audio(preview_url: str, output_path: str, timeout: int = config.TIMEOUT) -> bool:
    """
    Download audio from preview URL.

    Args:
        preview_url: URL to audio file
        output_path: Path to save audio file
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(preview_url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.debug(f"Downloaded audio to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download audio from {preview_url}: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Test Spotify source
    source = SpotifySource()

    # Test track metadata
    print("\n=== Testing track metadata ===")
    track = source.get_track_metadata("3n3Ppam7vgaVa1iaRUc9Lp")  # Mr. Brightside by The Killers
    if track:
        print(f"Track: {track['title']} by {track['artist']}")
        print(f"Preview URL: {track['preview_url']}")

    # Test search
    print("\n=== Testing search ===")
    results = source.search_tracks("indie rock", limit=5)
    print(f"Found {len(results)} tracks")
    for i, track in enumerate(results[:3], 1):
        print(f"{i}. {track['title']} by {track['artist']}")

    # Test global top tracks
    print("\n=== Testing global top tracks ===")
    top_tracks = source.get_global_top_tracks(limit=10)
    print(f"Retrieved {len(top_tracks)} top track IDs")

    # Test genre tracks
    print("\n=== Testing genre tracks ===")
    genre_tracks = source.get_tracks_by_genre("indie", limit=5)
    print(f"Retrieved {len(genre_tracks)} indie track IDs")
