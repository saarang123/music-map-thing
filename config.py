"""
Configuration file for the Video Generation Map project.
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

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# CLIP model configuration (for image + text embeddings)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # 768-dim embeddings
# Alternative: "openai/clip-vit-base-patch32" for faster, 512-dim

# Embedding configuration
IMAGE_EMBEDDING_DIM = 768  # CLIP image embedding
TEXT_EMBEDDING_DIM = 768   # CLIP text embedding
EMBEDDING_DIM = 768        # Combined embedding dimension (averaged or concatenated)
COMBINE_METHOD = "average" # "average" or "concat" (concat = 1536-dim)

# UMAP configuration
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC = 'euclidean'
UMAP_RANDOM_STATE = 42
UMAP_MODEL_PATH = MODELS_DIR / "umap_model.pkl"

# Database configuration
DATABASE_PATH = DATA_DIR / "embeddings.db"
BASE_GENERATIONS_PATH = DATA_DIR / "base_generations.json"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)
IMAGES_CACHE_DIR = DATA_DIR / "images_cache"
IMAGES_CACHE_DIR.mkdir(exist_ok=True)

# Generation collection targets
TARGET_TOTAL_GENERATIONS = 100000  # 50K-100K+ generations
TARGET_PER_CATEGORY = 25000        # ~25K per category

# Video generation categories
CONTENT_CATEGORIES = [
    "fantasy",
    "realistic",
    "cinematic",
    "animation"
]

# Supabase table configuration
SUPABASE_TABLE_NAME = "generations"  # Default table name, can be overridden
SUPABASE_IMAGE_COLUMN = "image_url"  # Column containing image URL/path
SUPABASE_PROMPT_COLUMN = "prompt"    # Column containing prompt text
SUPABASE_CATEGORY_COLUMN = "category"  # Column containing category
SUPABASE_ID_COLUMN = "id"            # Primary key column
SUPABASE_USER_COLUMN = "user_id"     # User ID column (optional)
SUPABASE_TIMESTAMP_COLUMN = "created_at"  # Timestamp column (optional)

# Processing configuration
BATCH_SIZE = 32  # For embedding generation
CHECKPOINT_INTERVAL = 1000  # Save progress every N generations
MAX_RETRIES = 3  # For database/API calls
TIMEOUT = 30  # Seconds for requests

# Image processing
IMAGE_MAX_SIZE = (512, 512)  # Resize images for faster processing
KEEP_IMAGE_CACHE = False  # Delete downloaded images after processing to save space
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]  # Supported formats

# Visualization configuration
VIZ_OUTPUT_DIR = DATA_DIR / "visualizations"
VIZ_OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DPI = 300
VIZ_FIGSIZE = (12, 10)
VIZ_CMAP = "viridis"  # Color map for temporal gradient

# Visualization configuration - Category colors
CATEGORY_COLORS = {
    "fantasy": "#9B59B6",      # Purple
    "realistic": "#3498DB",    # Blue
    "cinematic": "#E74C3C",    # Red
    "animation": "#2ECC71",    # Green
    "unknown": "#95A5A6"       # Gray
}

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = DATA_DIR / "video_map.log"
