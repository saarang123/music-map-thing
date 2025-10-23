# Music Map Visualization

A music visualization system that maps your listening history onto a 2D "music universe" based on audio embeddings. Think of it as a daily/weekly/monthly Snapchat-style heat map showing your musical journey through different genres and vibes.

![Music Map Example](docs/example_map.png)

## Overview

This project creates a stable 2D coordinate system representing the "music universe" where similar-sounding songs are close together. Your listening history is then plotted as a path through this universe, showing your musical journey over time.

### Key Features

- **Audio-based similarity**: Uses MERT (Music Understanding Model) to analyze actual audio content
- **Stable coordinate system**: Everyone's map uses the same coordinates, enabling social comparison
- **Temporal visualization**: See your listening journey as a path colored by time
- **Interactive exploration**: HTML visualizations with hover info for each track
- **Modular design**: Easy to swap Spotify for Apple Music or other services

## Technical Stack

- **Audio Embeddings**: MERT-v1-330M (768-dimensional)
- **Dimensionality Reduction**: UMAP (2D projection)
- **Database**: Simple file-based storage (easily upgradeable to vector DB)
- **Music API**: Spotify (modular for easy swapping)
- **Visualization**: Matplotlib (static) + Plotly (interactive)

## Project Structure

```
music-map-thing/
├── config.py                 # Configuration and settings
├── requirements.txt          # Python dependencies
│
├── setup/                    # Pipeline 1: Setup & Base Map
│   ├── audio_source.py      # Abstract audio source + Spotify impl
│   ├── scrape_songs.py      # Collect 100K diverse songs
│   ├── generate_embeddings.py  # MERT embedding generation
│   ├── train_umap.py        # Train UMAP model
│   └── database.py          # Database operations
│
├── user/                     # Pipeline 2: User Visualization
│   ├── user_history.py      # Fetch listening history
│   ├── embed_user_tracks.py # Get/compute track embeddings
│   ├── visualize.py         # Generate visualizations
│   └── main.py              # End-to-end pipeline
│
├── models/                   # Saved models
│   └── umap_model.pkl       # Fitted UMAP (after setup)
│
└── data/                     # Data storage
    ├── base_songs.json      # 100K song list
    ├── embeddings_cache/    # Embedding files
    └── visualizations/      # Output visualizations
```

## Setup

### 1. Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- Spotify Developer Account

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Spotify API Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Copy the Client ID and Client Secret
4. Set redirect URI to `http://localhost:8888/callback`

Create a `.env` file:

```bash
cp .env.template .env
```

Edit `.env` and add your credentials:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
```

## Usage

### Pipeline 1: Setup Base Map (One-time)

This creates the stable "music universe" that all users map to.

#### Step 1: Collect Songs (Recommended: Start with 1K for testing)

```bash
# Collect a small subset for testing (1K songs, ~5-10 minutes)
python setup/scrape_songs.py --subset 1000

# For production: collect full 100K songs (~2-3 hours)
python setup/scrape_songs.py
```

#### Step 2: Generate Embeddings

```bash
# This processes audio and generates MERT embeddings
# 1K songs: ~10-20 minutes on GPU, ~1-2 hours on CPU
# 100K songs: ~20-30 minutes on 8x H100, ~2-3 hours on single GPU

python setup/generate_embeddings.py --input data/base_songs.json

# To process a single track (for testing):
python setup/generate_embeddings.py --single-track 3n3Ppam7vgaVa1iaRUc9Lp
```

#### Step 3: Train UMAP Model

```bash
# Fit UMAP model on embeddings (~5-10 minutes for 1K, ~30-60 minutes for 100K)
python setup/train_umap.py

# This creates models/umap_model.pkl and saves 2D coordinates
```

### Pipeline 2: Visualize Your Music (Daily use)

Once the base map is set up, generate your personal music map:

```bash
# Generate music map for last week (default)
python user/main.py

# Generate map for last month
python user/main.py --timeframe month

# Generate only interactive HTML
python user/main.py --format plotly

# Custom output directory
python user/main.py --output ./my_maps
```

**Note**: The first time you run this, you'll need to authenticate with Spotify. A browser window will open for OAuth login.

### Output

The pipeline generates:
- `listening_history.json` - Raw listening data
- `tracks_with_coords.json` - Tracks with 2D coordinates
- `music_map.png` - Static visualization
- `music_map.html` - Interactive visualization (open in browser!)

## Advanced Usage

### Individual Components

#### Fetch Listening History Only

```bash
python user/user_history.py --timeframe week --stats
```

#### Get Embeddings for Specific Tracks

```bash
python user/embed_user_tracks.py --tracks TRACK_ID_1 TRACK_ID_2
```

#### Create Visualization from Existing Data

```bash
python user/visualize.py --tracks-file data/tracks_with_coords.json --format both
```

### Configuration

Edit `config.py` to customize:

- UMAP parameters (`UMAP_N_NEIGHBORS`, `UMAP_MIN_DIST`)
- Visualization settings (`VIZ_FIGSIZE`, `VIZ_CMAP`)
- Song collection targets
- Processing batch sizes

### Testing on Subset

For quick testing and development:

```bash
# 1. Collect 1K songs
python setup/scrape_songs.py --subset 1000

# 2. Generate embeddings
python setup/generate_embeddings.py

# 3. Train UMAP
python setup/train_umap.py

# 4. Generate your map
python user/main.py --timeframe week
```

## How It Works

### The Music Universe

1. **Diverse song collection**: Scrape 100K songs across genres, markets, and time periods
2. **Audio analysis**: Process each song through MERT to extract 768-dim embeddings
3. **Dimensionality reduction**: Use UMAP to project embeddings to 2D while preserving similarity
4. **Stable coordinates**: Save the fitted UMAP model for consistent projection

### Your Journey

1. **Fetch history**: Get your recently played tracks from Spotify
2. **Embed new tracks**: Generate embeddings for any tracks not in the base map
3. **Project to map**: Use `transform()` (not `fit_transform()`) to project onto the existing map
4. **Visualize path**: Create time-colored path showing your musical journey

## Understanding Your Map

- **Clusters**: Similar genres/styles naturally cluster together
- **Distance**: Closer points = more similar audio characteristics
- **Path**: Your listening journey through time
- **Colors**: Gradient shows temporal progression (older → recent)
- **Stars**: Green = start of timeframe, Red = most recent

## Troubleshooting

### No preview URL for tracks

Some Spotify tracks don't have 30s preview URLs. These tracks are skipped. This is normal and expected (~20-30% of tracks).

### CUDA out of memory

Reduce batch size in `config.py`:

```python
BATCH_SIZE = 16  # Default is 32
```

Or use CPU:

```bash
python setup/generate_embeddings.py --device cpu
```

### Spotify authentication issues

Make sure your redirect URI in Spotify Dashboard matches exactly:
```
http://localhost:8888/callback
```

Delete cached credentials if needed:
```bash
rm .cache-*
```

## Performance Tips

- **Use GPU**: 10-50x faster for embedding generation
- **Start small**: Test with 1K songs before scaling to 100K
- **Checkpointing**: Embedding generation saves progress every 1K songs
- **Parallel processing**: Run on 8x H100 node for production scale

## Extending the Project

### Add Apple Music Support

1. Create `AppleMusicSource` class in `audio_source.py`
2. Implement the `AudioSource` interface
3. Swap in initialization: `source = AppleMusicSource()`

### Upgrade to Vector Database

Replace `EmbeddingDatabase` in `database.py` with:
- Qdrant for scalable vector search
- Milvus for large-scale deployment
- PostgreSQL + pgvector for SQL + vectors

### Add Social Features

- Compare maps with friends (same coordinate system!)
- Find users with similar listening patterns
- Discover new music from nearby points on map

## Future Enhancements

- [ ] Web UI (React/Next.js)
- [ ] Animated path videos for social sharing
- [ ] Genre labels on map regions
- [ ] Heat maps for time spent in zones
- [ ] Playlist generation from map regions
- [ ] Real-time tracking and updates
- [ ] Mobile app

## Citations

**MERT Model**:
```
@inproceedings{li2023mert,
  title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author={Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Lin, Chenghao and Ragni, Anton and Benetos, Emmanouil and Gyenge, Norbert and others},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference},
  year={2023}
}
```

**UMAP**:
```
@article{mcinnes2018umap,
  title={UMAP: Uniform Manifold Approximation and Projection},
  author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
  journal={Journal of Open Source Software},
  year={2018}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For questions or issues:
- Open a GitHub issue
- Check existing issues for solutions
- Review the troubleshooting section

---

**Built with love for music visualization**
