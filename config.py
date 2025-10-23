"""
Configuration file for the Music Map project.
Store API keys and credentials in a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SETUP_DIR = PROJECT_ROOT / "setup"
USER_DIR = PROJECT_ROOT / "user"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

# MERT model configuration
MERT_MODEL_NAME = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24000
EMBEDDING_DIM = 768

# UMAP configuration
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC = 'euclidean'
UMAP_RANDOM_STATE = 42
UMAP_MODEL_PATH = MODELS_DIR / "umap_model.pkl"

# Database configuration
DATABASE_PATH = DATA_DIR / "embeddings.db"
BASE_SONGS_PATH = DATA_DIR / "base_songs.json"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)

# Song collection targets
TARGET_TOTAL_SONGS = 100000
TARGET_GLOBAL_TOP = 10000
TARGET_GENRE_DIVERSITY = 70000
TARGET_GEOGRAPHIC_DIVERSITY = 10000
TARGET_RECENT_RELEASES = 10000

# Spotify genres (125 genre seeds)
SPOTIFY_GENRES = [
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
    "black-metal", "bluegrass", "blues", "bossanova", "brazil", "breakbeat",
    "british", "cantopop", "chicago-house", "children", "chill", "classical",
    "club", "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
    "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep",
    "edm", "electro", "electronic", "emo", "folk", "forro", "french", "funk",
    "garage", "german", "gospel", "goth", "grindcore", "groove", "grunge",
    "guitar", "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal",
    "hip-hop", "holidays", "honky-tonk", "house", "idm", "indian", "indie",
    "indie-pop", "industrial", "iranian", "j-dance", "j-idol", "j-pop", "j-rock",
    "jazz", "k-pop", "kids", "latin", "latino", "malay", "mandopop", "metal",
    "metal-misc", "metalcore", "minimal-techno", "movies", "mpb", "new-age",
    "new-release", "opera", "pagode", "party", "philippines-opm", "piano",
    "pop", "pop-film", "post-dubstep", "power-pop", "progressive-house",
    "psych-rock", "punk", "punk-rock", "r-n-b", "rainy-day", "reggae",
    "reggaeton", "road-trip", "rock", "rock-n-roll", "rockabilly", "romance",
    "sad", "salsa", "samba", "sertanejo", "show-tunes", "singer-songwriter",
    "ska", "sleep", "songwriter", "soul", "soundtracks", "spanish", "study",
    "summer", "swedish", "synth-pop", "tango", "techno", "trance", "trip-hop",
    "turkish", "work-out", "world-music"
]

# Geographic markets for diversity
GEOGRAPHIC_MARKETS = [
    "US", "GB", "BR", "IN", "KR", "JP", "MX", "DE", "FR", "IT",
    "ES", "CA", "AU", "NL", "SE", "NO", "PL", "AR", "CL", "CO"
]

# Processing configuration
BATCH_SIZE = 32  # For embedding generation
CHECKPOINT_INTERVAL = 1000  # Save progress every N songs
MAX_RETRIES = 3  # For API calls
TIMEOUT = 30  # Seconds for API requests

# Audio processing
AUDIO_DURATION = 30  # seconds
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(exist_ok=True)
KEEP_AUDIO_CACHE = False  # Delete audio files after processing to save space

# Visualization configuration
VIZ_OUTPUT_DIR = DATA_DIR / "visualizations"
VIZ_OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DPI = 300
VIZ_FIGSIZE = (12, 10)
VIZ_CMAP = "viridis"  # Color map for temporal gradient

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = DATA_DIR / "music_map.log"
