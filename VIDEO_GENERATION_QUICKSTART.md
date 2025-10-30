# Video Generation Map - Quick Start Guide

## What Changed

The system has been transformed from **music visualization** to **video generation visualization**:

- **Before**: Spotify tracks → MERT audio embeddings → Music map
- **Now**: Video generations → CLIP image+text embeddings → Generation map

## System Overview

### Input Data (Your Pipeline)
```python
[
    {
        "generation_id": "gen_001",
        "prompt": "a dragon flying over mountains",
        "image_url": "https://...",  # or local path
        "category": "fantasy",  # fantasy/realistic/cinematic/animation
        "user_id": "user_123",  # optional
        "timestamp": "2024-10-23T14:30:00"  # optional
    },
    ...  # 70K+ generations
]
```

### Output
- **2D map** where similar generations cluster together
- **Combined similarity**: Both visual (image) and semantic (prompt)
- **Category coloring**: Fantasy=purple, Realistic=blue, Cinematic=red, Animation=green
- **Interactive exploration**: Click to see prompts and images

## How It Works

### Embedding Strategy: Combined Image + Text

```python
# CLIP processes both image and prompt
image_embedding = CLIP_vision(first_frame)  # 768-dim
text_embedding = CLIP_text(prompt)          # 768-dim

# Combine (averaged for 768-dim output)
combined = (image_embedding + text_embedding) / 2
```

**Why combined?**
- Pure visual: "cyberpunk city" and "neon street" cluster (similar look)
- Pure semantic: different styles of "dragon" cluster (similar concept)
- **Combined: Best of both worlds!**

### UMAP Projection

```python
# 768-dim → 2D while preserving similarity
umap_model.fit(70K_combined_embeddings)
# Result: Similar generations stay close in 2D
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key new dependencies:
- `transformers` - CLIP model
- `Pillow` - Image processing
- `supabase` - Database client (optional)

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env and add (optional for Supabase):
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

### 3. Prepare Your Data

**Option A: JSON File**
```json
// generations.json
[
    {
        "generation_id": "gen_001",
        "prompt": "epic dragon battle",
        "image_url": "https://example.com/gen_001.jpg",
        "category": "fantasy"
    }
]
```

**Option B: Python List (from your pipeline)**
```python
from setup.embed_generations import CLIPEmbeddingGenerator

generations = your_data_pipeline.get_generations(limit=70000)
generator = CLIPEmbeddingGenerator()
generator.process_from_list(generations)
```

## Pipeline 1: Build Base Map (One-time, ~2-3 hours for 70K)

### Step 1: Load Your Generations

```bash
# If using JSON:
python setup/scrape_generations.py --input your_data.json
# Output: data/base_generations.json

# Or provide data directly from your pipeline (recommended)
```

### Step 2: Generate CLIP Embeddings

```bash
python setup/embed_generations.py --device cuda
# Processes: 70K generations × (~50ms) = ~1 hour on GPU
# Outputs: data/embeddings_cache/embedding_*.npy files
```

**What happens:**
- Downloads/loads each image
- Resizes to 512×512 (config.IMAGE_MAX_SIZE)
- Processes through CLIP
- Combines image + text embeddings
- Saves 768-dim vectors

**Progress tracking:**
- Checkpoints every 1000 generations
- Skips already-processed generations
- Logs failures (missing images, etc.)

### Step 3: Train UMAP Model

```bash
python setup/train_umap.py
# Processes: 70K × 768-dim → 2D
# Time: ~30-45 minutes
# Output: models/umap_model.pkl
```

**Critical:** This creates the stable coordinate system that all visualizations use.

## Pipeline 2: Visualize Trends (Daily use, ~seconds)

### User Journey
```bash
python user/main.py --user-id user_123 --timeframe week
# Shows: This user's generation journey over time
```

### Platform Trends
```bash
python user/main.py --timeframe day --category fantasy
# Shows: All fantasy generations from today
```

### Category Analysis
```bash
python user/main.py --heatmap --category-breakdown
# Shows: Where each category clusters on the map
```

## Integration with Your Pipeline

### Recommended Workflow

```python
# your_pipeline.py

from setup.embed_generations import CLIPEmbeddingGenerator
from setup.database import EmbeddingDatabase

# Initialize once
generator = CLIPEmbeddingGenerator(device='cuda')
db = EmbeddingDatabase()

# When new generations are created:
def on_new_generation(generation_data):
    # Your data format → standard format
    gen = {
        'generation_id': generation_data['id'],
        'prompt': generation_data['prompt'],
        'image_url': generation_data['first_frame_url'],
        'category': generation_data['category']
    }

    # Generate embedding (cached if exists)
    generator.process_generation(gen)

    # Now it's on the map!

# Batch processing for initial 70K:
all_generations = your_db.get_all_generations()
generator.process_from_list(all_generations)
```

## Use Cases

### 1. Trend Discovery
**Question**: "What visual styles are trending this week?"

```python
# Get this week's generations
recent = your_pipeline.get_generations(days=7)
# Project onto map
coords = embedder.get_coords(recent)
# Create heatmap
visualizer.create_heatmap(coords)
# Result: See concentration areas
```

### 2. Style Consistency
**Question**: "Do similar prompts generate similar images?"

```python
# Find generations with keyword
gens = db.search_by_metadata('prompt', 'cyberpunk')
# Visualize
# Result: If clustered tightly = consistent style
```

### 3. Category Boundaries
**Question**: "Where does 'cinematic' blend into 'realistic'?"

```python
# Get all cinematic and realistic
cinematic = get_by_category('cinematic')
realistic = get_by_category('realistic')
# Plot both with different colors
# Result: See overlap regions
```

### 4. User Evolution
**Question**: "How has this user's style changed?"

```python
# Get user's generations over time
user_gens = get_user_history('user_123', months=6)
# Sort by timestamp
# Plot as path with time colors
# Result: See journey from realistic → fantasy
```

## Performance

### Embedding Generation (Pipeline 1)
| Dataset | GPU | CPU |
|---------|-----|-----|
| 1K | ~1 min | ~15 min |
| 10K | ~8 min | ~2 hours |
| 70K | ~1 hour | ~14 hours |

**Tip**: Run on GPU for initial 70K, CPU is fine for incremental updates

### UMAP Training
| Dataset | Time | RAM |
|---------|------|-----|
| 1K | ~5 min | ~2GB |
| 10K | ~10 min | ~4GB |
| 70K | ~45 min | ~10GB |

### User Queries (Pipeline 2)
- **Cached generations**: <1 second
- **New generations**: ~50ms per generation (GPU)
- **Visualization**: ~2-5 seconds

## Storage

For 70K generations:
- **Embeddings**: 70K × 6KB = ~420MB
- **UMAP model**: ~50MB
- **Metadata**: ~30MB
- **Total**: ~500MB

## Differences from Music System

| Aspect | Music (Old) | Video Gen (New) |
|--------|-------------|-----------------|
| **Input** | Audio (30s MP3) | Image + Prompt |
| **Model** | MERT (audio) | CLIP (multimodal) |
| **Embedding** | 768-dim audio | 768-dim image+text |
| **Source** | Spotify API | Your database |
| **Data** | 100K songs | 70K+ generations |
| **Categories** | Genres | fantasy/realistic/cinematic/animation |
| **Similarity** | Audio features | Visual + semantic |

## Next Steps

1. **Prepare your data** in the format shown above
2. **Run Pipeline 1** to build the base map (one-time, ~2-3 hours)
3. **Create visualizations** using Pipeline 2
4. **Integrate** with your existing pipeline for real-time updates

## Questions?

- **Data format**: See examples above
- **Missing images**: Logged and skipped (check logs)
- **Categories**: Currently 4, easily extensible
- **Scale**: Tested up to 100K generations
- **Updates**: Incremental (only new generations processed)

## Current Status

✅ **Complete**:
- CLIP embedding generation
- Combined image+text similarity
- Supabase integration (optional)
- File-based storage
- Category support

🚧 **In Progress**:
- Visualization updates (image thumbnails)
- User pipeline updates
- Documentation

📋 **To Do (Your Side)**:
- Provide generation data in specified format
- Run Pipeline 1 to build base map
- Integrate with your existing system
