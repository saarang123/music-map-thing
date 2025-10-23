"""
Database operations for storing and retrieving song embeddings and metadata.
Uses simple file-based storage for MVP (JSON + numpy files).
Can be upgraded to vector database (Qdrant, Milvus, PostgreSQL+pgvector) later.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class EmbeddingDatabase:
    """
    File-based embedding database.
    Structure:
        - embeddings_cache/embeddings_{track_id}.npy - 768-dim embedding
        - embeddings_cache/metadata.json - track metadata
        - embeddings_cache/umap_coords.json - 2D UMAP coordinates
        - embeddings_cache/index.json - mapping of track IDs to file paths
    """

    def __init__(self, cache_dir: Path = config.EMBEDDINGS_CACHE_DIR):
        """Initialize the database."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.umap_coords_file = self.cache_dir / "umap_coords.json"
        self.index_file = self.cache_dir / "index.json"

        # Load or initialize metadata
        self.metadata = self._load_json(self.metadata_file, default={})
        self.umap_coords = self._load_json(self.umap_coords_file, default={})
        self.index = self._load_json(self.index_file, default={})

        logger.info(f"EmbeddingDatabase initialized with {len(self.index)} tracks")

    def _load_json(self, file_path: Path, default: dict = None) -> dict:
        """Load JSON file or return default if not exists."""
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                return default if default is not None else {}
        return default if default is not None else {}

    def _save_json(self, data: dict, file_path: Path):
        """Save data to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            raise

    def _get_embedding_path(self, track_id: str) -> Path:
        """Get the file path for a track's embedding."""
        return self.cache_dir / f"embedding_{track_id}.npy"

    def add_embedding(self, track_id: str, embedding: np.ndarray,
                     metadata: Optional[Dict] = None,
                     umap_coords: Optional[Tuple[float, float]] = None):
        """
        Add or update an embedding and its metadata.

        Args:
            track_id: Unique track identifier
            embedding: 768-dimensional embedding vector
            metadata: Optional track metadata (title, artist, etc.)
            umap_coords: Optional 2D UMAP coordinates (x, y)
        """
        if embedding.shape != (config.EMBEDDING_DIM,):
            raise ValueError(f"Embedding must be {config.EMBEDDING_DIM}-dimensional, got {embedding.shape}")

        # Save embedding
        embedding_path = self._get_embedding_path(track_id)
        try:
            np.save(embedding_path, embedding)
            self.index[track_id] = str(embedding_path)
        except Exception as e:
            logger.error(f"Failed to save embedding for {track_id}: {e}")
            raise

        # Save metadata if provided
        if metadata:
            self.metadata[track_id] = {
                **metadata,
                'created_at': datetime.now().isoformat()
            }

        # Save UMAP coordinates if provided
        if umap_coords:
            self.umap_coords[track_id] = {
                'x': float(umap_coords[0]),
                'y': float(umap_coords[1])
            }

        logger.debug(f"Added embedding for track {track_id}")

    def get_embedding(self, track_id: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding by track ID.

        Returns:
            768-dimensional numpy array or None if not found
        """
        if track_id not in self.index:
            return None

        try:
            embedding_path = Path(self.index[track_id])
            if not embedding_path.exists():
                logger.warning(f"Embedding file not found for {track_id}")
                return None

            embedding = np.load(embedding_path)
            return embedding
        except Exception as e:
            logger.error(f"Failed to load embedding for {track_id}: {e}")
            return None

    def get_embeddings(self, track_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Retrieve multiple embeddings.

        Returns:
            Dict mapping track_id to embedding (only for found tracks)
        """
        embeddings = {}
        for track_id in track_ids:
            embedding = self.get_embedding(track_id)
            if embedding is not None:
                embeddings[track_id] = embedding

        return embeddings

    def get_metadata(self, track_id: str) -> Optional[Dict]:
        """Get metadata for a track."""
        return self.metadata.get(track_id)

    def get_umap_coords(self, track_id: str) -> Optional[Tuple[float, float]]:
        """Get UMAP coordinates for a track."""
        coords = self.umap_coords.get(track_id)
        if coords:
            return (coords['x'], coords['y'])
        return None

    def has_embedding(self, track_id: str) -> bool:
        """Check if an embedding exists for a track."""
        return track_id in self.index

    def batch_add_embeddings(self, embeddings_data: List[Dict]):
        """
        Add multiple embeddings in batch.

        Args:
            embeddings_data: List of dicts with keys: track_id, embedding, metadata, umap_coords
        """
        for data in embeddings_data:
            self.add_embedding(
                track_id=data['track_id'],
                embedding=data['embedding'],
                metadata=data.get('metadata'),
                umap_coords=data.get('umap_coords')
            )

        # Save all metadata and coordinates at once
        self._save_json(self.index, self.index_file)
        self._save_json(self.metadata, self.metadata_file)
        self._save_json(self.umap_coords, self.umap_coords_file)

        logger.info(f"Batch added {len(embeddings_data)} embeddings")

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all embeddings as a matrix.

        Returns:
            Tuple of (track_ids, embeddings_matrix)
            where embeddings_matrix is N x 768
        """
        track_ids = list(self.index.keys())
        embeddings = []

        for track_id in track_ids:
            embedding = self.get_embedding(track_id)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Skipping missing embedding for {track_id}")

        embeddings_matrix = np.array(embeddings)
        logger.info(f"Loaded {embeddings_matrix.shape[0]} embeddings")

        return track_ids, embeddings_matrix

    def update_umap_coords(self, coords_dict: Dict[str, Tuple[float, float]]):
        """
        Update UMAP coordinates for multiple tracks.

        Args:
            coords_dict: Dict mapping track_id to (x, y) tuple
        """
        for track_id, coords in coords_dict.items():
            self.umap_coords[track_id] = {
                'x': float(coords[0]),
                'y': float(coords[1])
            }

        self._save_json(self.umap_coords, self.umap_coords_file)
        logger.info(f"Updated UMAP coordinates for {len(coords_dict)} tracks")

    def save(self):
        """Save all data to disk."""
        self._save_json(self.index, self.index_file)
        self._save_json(self.metadata, self.metadata_file)
        self._save_json(self.umap_coords, self.umap_coords_file)
        logger.info("Database saved to disk")

    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_tracks': len(self.index),
            'tracks_with_metadata': len(self.metadata),
            'tracks_with_umap_coords': len(self.umap_coords),
            'cache_dir': str(self.cache_dir)
        }

    def search_by_metadata(self, key: str, value: str) -> List[str]:
        """
        Search tracks by metadata field.

        Args:
            key: Metadata key (e.g., 'artist', 'title')
            value: Value to search for (case-insensitive partial match)

        Returns:
            List of matching track IDs
        """
        matching_ids = []
        value_lower = value.lower()

        for track_id, metadata in self.metadata.items():
            if key in metadata:
                field_value = str(metadata[key]).lower()
                if value_lower in field_value:
                    matching_ids.append(track_id)

        return matching_ids

    def delete_track(self, track_id: str):
        """Delete a track and all its data."""
        # Delete embedding file
        if track_id in self.index:
            embedding_path = Path(self.index[track_id])
            if embedding_path.exists():
                embedding_path.unlink()
            del self.index[track_id]

        # Delete metadata and coords
        self.metadata.pop(track_id, None)
        self.umap_coords.pop(track_id, None)

        logger.info(f"Deleted track {track_id}")

    def clear_all(self):
        """Clear all data (use with caution!)."""
        # Delete all embedding files
        for embedding_path in self.cache_dir.glob("embedding_*.npy"):
            embedding_path.unlink()

        # Clear all dictionaries
        self.index = {}
        self.metadata = {}
        self.umap_coords = {}

        # Save empty files
        self.save()

        logger.warning("Database cleared")


# Example usage
if __name__ == "__main__":
    # Create database
    db = EmbeddingDatabase()

    # Test adding an embedding
    print("\n=== Testing add embedding ===")
    test_embedding = np.random.randn(config.EMBEDDING_DIM)
    test_metadata = {
        'track_id': 'test_track_123',
        'title': 'Test Song',
        'artist': 'Test Artist',
        'preview_url': 'http://example.com/preview.mp3'
    }

    db.add_embedding(
        track_id='test_track_123',
        embedding=test_embedding,
        metadata=test_metadata,
        umap_coords=(0.5, 0.7)
    )

    # Test retrieving
    print("\n=== Testing get embedding ===")
    retrieved = db.get_embedding('test_track_123')
    print(f"Retrieved embedding shape: {retrieved.shape}")
    print(f"Embeddings match: {np.allclose(test_embedding, retrieved)}")

    # Test metadata
    print("\n=== Testing get metadata ===")
    metadata = db.get_metadata('test_track_123')
    print(f"Metadata: {metadata}")

    # Test UMAP coords
    print("\n=== Testing get UMAP coords ===")
    coords = db.get_umap_coords('test_track_123')
    print(f"UMAP coords: {coords}")

    # Test stats
    print("\n=== Database stats ===")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Save database
    db.save()
    print("\n=== Database saved ===")

    # Clean up test data
    db.delete_track('test_track_123')
    print("Test track deleted")
