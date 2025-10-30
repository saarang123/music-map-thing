# Pipeline Documentation

## Overview

The Music Map system consists of two main pipelines:
1. **Pipeline 1**: Setup & Base Map Creation (one-time, ~2-4 hours for 100K songs)
2. **Pipeline 2**: User Visualization (daily use, ~1-5 minutes)

---

## Pipeline 1: Setup & Base Map Creation

### Purpose
Create a stable "music universe" coordinate system that all users map onto.

### Steps

#### 1.1 Song Collection (`scrape_songs.py`)

**What it does**:
- Collects 100K diverse songs from Spotify
- Ensures coverage across genres, markets, and time periods

**How it works**:
```python
scraper = SongScraper()
├── collect_global_top_tracks(10K)
│   └── Uses Spotify Global Top 50 + popular searches
├── collect_genre_diversity(70K)
│   └── ~560 songs per genre × 125 genres
├── collect_geographic_diversity(10K)
│   └── ~500 songs per market × 20 markets
└── collect_recent_releases(10K)
    └── New music from last 6 months
```

**Input**: Spotify API credentials
**Output**: `data/base_songs.json`

```json
{
  "total_tracks": 100000,
  "tracks": {
    "track_id_1": {
      "track_id": "...",
      "title": "Mr. Brightside",
      "artist": "The Killers",
      "preview_url": "https://...",
      "source": "global_top",
      "popularity": 88
    }
  },
  "statistics": {
    "global_top": 10000,
    "genre_diversity": 70000,
    "geographic_diversity": 10000,
    "recent_releases": 10000
  }
}
```

**Parallelization**: Sequential (rate-limited by Spotify API)
**Time**: ~1-2 hours for 100K songs

---

#### 1.2 Embedding Generation (`generate_embeddings.py`)

**What it does**:
- Downloads 30s audio previews
- Generates 768-dim MERT embeddings
- Stores embeddings + metadata

**How it works**:
```python
generator = MERTEmbeddingGenerator(device='cuda')

For each track in base_songs.json:
├── Check if embedding exists (db.has_embedding)
│   └── Skip if yes (avoid reprocessing)
├── Download 30s preview to temp file
├── Load and preprocess audio
│   ├── Resample to 24kHz
│   └── Convert to mono
├── Generate MERT embedding
│   ├── Process through MERT-v1-330M
│   ├── Extract last hidden state
│   └── Mean pool over time → 768-dim vector
├── Save to database
│   ├── embedding_TRACK_ID.npy
│   ├── Update metadata.json
│   └── Update index.json
└── Checkpoint every 1K songs
```

**Input**: `data/base_songs.json`
**Output**:
- `data/embeddings_cache/embedding_*.npy` (100K files)
- `data/embeddings_cache/metadata.json`
- `data/embeddings_cache/index.json`

**Parallelization**:
- ⚠️ **Currently sequential** (one track at a time)
- 🚀 **Opportunity**: Batch processing (see optimization section)

**Time**:
- 8x H100: ~20-30 minutes
- 1x GPU: ~2-3 hours
- CPU: ~10-15 hours

**Failure handling**:
- Tracks without preview URLs: logged and skipped (~20-30%)
- Audio download failures: retried, then skipped
- MERT failures: logged and skipped
- Checkpoints every 1K songs for fault tolerance

---

#### 1.3 UMAP Training (`train_umap.py`)

**What it does**:
- Loads all 100K embeddings
- Fits UMAP model to reduce 768D → 2D
- Saves fitted model for consistent future projections

**How it works**:
```python
trainer = UMAPTrainer()

1. Load embeddings
   ├── Read index.json
   ├── Load all embedding_*.npy files
   └── Stack into matrix [100K × 768]

2. Fit UMAP
   ├── Initialize UMAP(n_neighbors=15, min_dist=0.1)
   ├── Fit and transform embeddings
   └── Result: [100K × 2] coordinate matrix

3. Save outputs
   ├── models/umap_model.pkl (fitted model)
   ├── Update umap_coords.json (2D coordinates)
   └── Create visualization (music_universe_base.png)
```

**Input**: All `embedding_*.npy` files
**Output**:
- `models/umap_model.pkl` (CRITICAL: used for user projections)
- `data/embeddings_cache/umap_coords.json`
- `data/visualizations/music_universe_base.png`

**Parallelization**:
- ✅ **Already parallelized** (UMAP uses multiple cores internally)
- Can tune with `n_jobs` parameter

**Time**:
- 100K songs: ~30-60 minutes
- 1K songs: ~5-10 minutes

**Critical Note**:
```python
# CORRECT: Fit once during setup
coords = umap_model.fit_transform(base_embeddings)
joblib.dump(umap_model, 'umap_model.pkl')

# Later, for new songs
umap_model = joblib.load('umap_model.pkl')
new_coords = umap_model.transform(new_embeddings)  # NOT fit_transform!
```

---

## Pipeline 2: User Visualization

### Purpose
Generate personalized music map for a user's listening history.

### Steps

#### 2.1 Fetch Listening History (`user_history.py`)

**What it does**:
- Authenticates user with Spotify OAuth
- Fetches recently played tracks
- Retrieves metadata

**How it works**:
```python
fetcher = UserHistoryFetcher()

1. OAuth authentication
   ├── Opens browser for user login
   ├── Gets access token
   └── Saves to .cache-USERNAME

2. Fetch history
   ├── Call /me/player/recently-played
   ├── Filter by timeframe (day/week/month)
   └── Returns [(track_id, timestamp), ...]

3. Get metadata
   ├── Batch fetch track info (50 at a time)
   └── Returns full track details
```

**Input**: User authorization
**Output**: List of tracks with timestamps

```python
[
  {
    "track_id": "abc123",
    "timestamp": "2024-10-23T14:30:00",
    "title": "Mr. Brightside",
    "artist": "The Killers",
    "preview_url": "https://...",
    "popularity": 88
  }
]
```

**Parallelization**: Sequential (API limited to 50 tracks/request)
**Time**: ~5-10 seconds

---

#### 2.2 Embed User Tracks (`embed_user_tracks.py`)

**What it does**:
- Gets embeddings for user's tracks
- Generates embeddings for missing tracks
- Projects to 2D coordinates

**How it works**:
```python
embedder = UserTrackEmbedder(device='cuda')

For each track_id in user history:
├── Check database
│   ├── If exists: load embedding_TRACK_ID.npy
│   └── If missing: generate embedding (lazy init MERT)
├── Collect all embeddings
└── Project to 2D
    ├── Load models/umap_model.pkl
    ├── Transform (not fit!) to 2D
    └── Return {track_id: (x, y)}
```

**Input**: List of track IDs
**Output**: Dict of {track_id: (x, y)}

**Parallelization**:
- ⚠️ **Currently sequential** for new tracks
- ✅ **Fast path**: Existing embeddings loaded in parallel
- 🚀 **Opportunity**: Batch new embeddings (see optimization)

**Time**:
- All tracks cached: ~1-2 seconds
- 10 new tracks: ~30-60 seconds (GPU)
- 50 new tracks: ~2-3 minutes (GPU)

---

#### 2.3 Visualization (`visualize.py`)

**What it does**:
- Creates static and interactive visualizations
- Shows listening journey as time-colored path

**How it works**:
```python
viz = MusicMapVisualizer()

1. Prepare data
   ├── Sort tracks by timestamp
   ├── Extract x, y coordinates
   └── Create color gradient (time-based)

2. Create matplotlib plot
   ├── Draw path with gradient line
   ├── Scatter points colored by time
   ├── Highlight start (green star) and end (red star)
   └── Save as PNG

3. Create plotly plot
   ├── Add interactive scatter plot
   ├── Add hover labels (title, artist, time)
   ├── Add path line
   └── Save as HTML
```

**Input**: List of tracks with coordinates
**Output**:
- `music_map.png` (static)
- `music_map.html` (interactive)

**Parallelization**:
- ✅ **Can parallelize**: Generate PNG and HTML simultaneously
- Currently sequential (fast enough)

**Time**: ~5-10 seconds

---

#### 2.4 End-to-End Pipeline (`main.py`)

**What it does**:
- Orchestrates entire user visualization pipeline
- Combines all steps into single command

**How it works**:
```python
pipeline = MusicMapPipeline(device='cuda')

pipeline.generate_music_map(timeframe='week'):
├── [1/4] Fetch listening history
│   └── ~5-10 seconds
├── [2/4] Get embeddings for tracks
│   └── ~1 second (cached) to ~3 minutes (new tracks)
├── [3/4] Project to 2D coordinates
│   └── ~1 second
└── [4/4] Create visualizations
    └── ~5-10 seconds

Total: ~15 seconds (cached) to ~5 minutes (many new tracks)
```

**Input**: User authentication + timeframe
**Output**: Complete visualization package

---

## Data Flow Diagram

```
Pipeline 1: Setup (One-time)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Spotify API                                                │
│      ↓                                                      │
│  scrape_songs.py → base_songs.json (100K tracks)           │
│      ↓                                                      │
│  generate_embeddings.py                                     │
│      ├→ Download audio previews                            │
│      ├→ MERT model (768-dim embeddings)                    │
│      └→ embeddings_cache/ (100K .npy files)                │
│      ↓                                                      │
│  train_umap.py                                              │
│      ├→ Load all embeddings                                │
│      ├→ Fit UMAP (768D → 2D)                               │
│      └→ umap_model.pkl + umap_coords.json                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Pipeline 2: User Visualization (Daily)
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Spotify API (OAuth)                                        │
│      ↓                                                      │
│  user_history.py → listening history (50-100 tracks)       │
│      ↓                                                      │
│  embed_user_tracks.py                                       │
│      ├→ Check cache (90% hit rate)                         │
│      ├→ Generate missing (10% new tracks)                  │
│      ├→ Load umap_model.pkl                                │
│      └→ Transform to 2D (same coordinate system!)          │
│      ↓                                                      │
│  visualize.py                                               │
│      ├→ matplotlib (PNG)                                   │
│      └→ plotly (interactive HTML)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Resource Requirements

### Pipeline 1 (Setup)

| Resource | Requirement | Notes |
|----------|-------------|-------|
| Storage | ~650MB for 100K songs | ~6KB per embedding |
| RAM | 8-16GB | 4GB for embeddings, 8GB for UMAP |
| GPU | 8GB+ VRAM | Optional but 10-50x faster |
| Network | 5-10GB download | 30s previews (~50KB each) |
| Time | 2-4 hours total | GPU recommended |

### Pipeline 2 (User)

| Resource | Requirement | Notes |
|----------|-------------|-------|
| Storage | +10MB per session | Visualizations + history |
| RAM | 2-4GB | Loads UMAP model + metadata |
| GPU | Optional | Only if generating new embeddings |
| Network | <1MB | Fetch listening history |
| Time | 15 seconds - 5 minutes | Depends on cache hit rate |

## Error Handling

### Pipeline 1

| Error | Handling | Recovery |
|-------|----------|----------|
| Missing preview URL | Skip track, log warning | Continue with next |
| Audio download fail | Retry 3x, then skip | Checkpoint preserves progress |
| MERT model crash | Log error, skip track | Manual inspection needed |
| Disk full | Stop, save checkpoint | Resume after freeing space |

### Pipeline 2

| Error | Handling | Recovery |
|-------|----------|----------|
| OAuth expired | Re-authenticate | Browser window opens |
| No listening history | Show warning, exit | User needs to play music |
| UMAP model missing | Clear error message | Run Pipeline 1 first |
| New track no preview | Skip from visualization | Note in output |

## Performance Characteristics

### Pipeline 1 Bottlenecks

1. **Network I/O**: Downloading 100K audio previews
   - Mitigation: Async downloads (future)

2. **MERT inference**: GPU compute time
   - Mitigation: Batch processing (see optimization)

3. **UMAP fitting**: CPU + RAM intensive
   - Mitigation: Use multiple cores (already done)

### Pipeline 2 Bottlenecks

1. **MERT for new tracks**: Rare but slow
   - Mitigation: Pre-populate cache with popular tracks

2. **UMAP model loading**: ~1-2 seconds
   - Mitigation: Keep in memory for web service

## Best Practices

### For Development/Testing

```bash
# Always start small!
python setup/scrape_songs.py --subset 1000
python setup/generate_embeddings.py
python setup/train_umap.py
python user/main.py --timeframe day
```

### For Production

```bash
# Full pipeline
python setup/scrape_songs.py  # 100K songs
python setup/generate_embeddings.py --device cuda
python setup/train_umap.py
python user/main.py --timeframe month
```

### For Debugging

```bash
# Test single track
python setup/generate_embeddings.py --single-track TRACK_ID

# Test setup
python test_setup.py

# Check logs
tail -f data/music_map.log
```

## Next Steps

See [PARALLELIZATION.md](PARALLELIZATION.md) for optimization opportunities.
