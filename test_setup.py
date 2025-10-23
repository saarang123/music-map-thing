#!/usr/bin/env python3
"""
Quick test script to verify setup is working correctly.
Tests basic functionality without requiring full data collection.
"""

import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")
    required_packages = [
        'torch',
        'torchaudio',
        'transformers',
        'numpy',
        'umap',
        'spotipy',
        'matplotlib',
        'plotly',
        'joblib'
    ]

    failed = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {package}")
        except ImportError as e:
            logger.error(f"  ✗ {package}: {e}")
            failed.append(package)

    if failed:
        logger.error(f"\nFailed to import: {', '.join(failed)}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    logger.info("All imports successful!\n")
    return True


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    try:
        import config
        logger.info(f"  ✓ Config loaded")
        logger.info(f"  ✓ Data dir: {config.DATA_DIR}")
        logger.info(f"  ✓ Models dir: {config.MODELS_DIR}")
        logger.info(f"  ✓ Embedding dim: {config.EMBEDDING_DIM}")

        if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
            logger.warning("  ⚠ Spotify credentials not set in .env file")
            logger.warning("    Copy .env.template to .env and add your credentials")
        else:
            logger.info("  ✓ Spotify credentials found")

        logger.info("")
        return True

    except Exception as e:
        logger.error(f"  ✗ Config error: {e}")
        return False


def test_modules():
    """Test that project modules can be imported."""
    logger.info("Testing project modules...")
    modules = [
        'setup.audio_source',
        'setup.database',
        'setup.scrape_songs',
        'setup.generate_embeddings',
        'setup.train_umap',
        'user.user_history',
        'user.embed_user_tracks',
        'user.visualize',
        'user.main'
    ]

    failed = []
    for module in modules:
        try:
            __import__(module)
            logger.info(f"  ✓ {module}")
        except Exception as e:
            logger.error(f"  ✗ {module}: {e}")
            failed.append(module)

    if failed:
        logger.error(f"\nFailed to import: {', '.join(failed)}")
        return False

    logger.info("All modules loaded successfully!\n")
    return True


def test_spotify_connection():
    """Test Spotify API connection."""
    logger.info("Testing Spotify connection...")
    try:
        import config
        if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
            logger.warning("  ⚠ Skipping (credentials not set)")
            logger.info("")
            return True

        from setup.audio_source import SpotifySource

        spotify = SpotifySource()
        # Try to get a well-known track
        track = spotify.get_track_metadata("3n3Ppam7vgaVa1iaRUc9Lp")  # Mr. Brightside

        if track:
            logger.info(f"  ✓ Connected to Spotify API")
            logger.info(f"  ✓ Test track: {track['title']} by {track['artist']}")
        else:
            logger.error("  ✗ Could not fetch track metadata")
            return False

        logger.info("")
        return True

    except Exception as e:
        logger.error(f"  ✗ Spotify connection failed: {e}")
        logger.info("")
        return False


def test_database():
    """Test database operations."""
    logger.info("Testing database...")
    try:
        from setup.database import EmbeddingDatabase
        import numpy as np
        import config

        db = EmbeddingDatabase()
        logger.info(f"  ✓ Database initialized")

        # Test adding and retrieving
        test_id = "test_track_xyz"
        test_embedding = np.random.randn(config.EMBEDDING_DIM)
        test_metadata = {'title': 'Test Track', 'artist': 'Test Artist'}

        db.add_embedding(test_id, test_embedding, metadata=test_metadata)
        retrieved = db.get_embedding(test_id)

        if retrieved is not None and np.allclose(test_embedding, retrieved):
            logger.info(f"  ✓ Add/retrieve operations work")
        else:
            logger.error(f"  ✗ Add/retrieve test failed")
            return False

        # Clean up
        db.delete_track(test_id)
        logger.info(f"  ✓ Delete operation works")

        stats = db.get_stats()
        logger.info(f"  ✓ Database stats: {stats['total_tracks']} tracks")

        logger.info("")
        return True

    except Exception as e:
        logger.error(f"  ✗ Database test failed: {e}")
        logger.info("")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Music Map Setup Test")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        ("Package imports", test_imports),
        ("Configuration", test_config),
        ("Project modules", test_modules),
        ("Spotify connection", test_spotify_connection),
        ("Database operations", test_database),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! Your setup is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Collect songs: python setup/scrape_songs.py --subset 1000")
        logger.info("2. Generate embeddings: python setup/generate_embeddings.py")
        logger.info("3. Train UMAP: python setup/train_umap.py")
        logger.info("4. Generate your map: python user/main.py")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
