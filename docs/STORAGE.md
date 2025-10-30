# Storage System Architecture

## Overview

The Music Map project uses a **file-based storage system** for the MVP. This is optimized for simplicity and fast development, with a clear upgrade path to vector databases.

## Storage Structure

```
data/
├── embeddings_cache/           # Embedding storage
│   ├── embedding_TRACK_ID.npy  # Individual 768-dim embeddings
│   ├── metadata.json           # Track metadata (title, artist, etc.)
│   ├── umap_coords.json        # 2D coordinates
│   └── index.json              # Track ID → file path mapping
│
├── base_songs.json             # 100K song list with metadata
│
├── visualizations/             # User-generated visualizations
│   └── session_TIMESTAMP/
│       ├── listening_history.json
│       ├── tracks_with_coords.json
│       ├── music_map.png
│       └── music_map.html
│
└── music_map.log               # Application logs
```

## File Formats

### 1. Embeddings (`.npy` files)

**File**: `data/embeddings_cache/embedding_{track_id}.npy`

- **Format**: NumPy binary format
- **Shape**: (768,) - 1D array
- **Dtype**: float32 or float64
- **Size**: ~6KB per file

**Storage strategy**:
- One file per track (enables parallel read/write)
- Fast loading with `np.load()` (~1-2ms per file)
- Easy to update individual tracks
- File system handles deduplication

**Pros**:
- Simple implementation
- Parallel-friendly (no file locking issues)
- Easy to backup/transfer
- No database overhead

**Cons**:
- Many small files (not ideal for very large scale)
- No built-in similarity search
- Requires custom indexing

### 2. Metadata (`metadata.json`)

**File**: `data/embeddings_cache/metadata.json`

```json
{
  "track_id_1": {
    "track_id": "3n3Ppam7vgaVa1iaRUc9Lp",
    "title": "Mr. Brightside",
    "artist": "The Killers",
    "preview_url": "https://...",
    "duration_ms": 222973,
    "popularity": 88,
    "created_at": "2024-10-23T12:34:56.789"
  },
  "track_id_2": { ... }
}
```

**Characteristics**:
- Loaded into memory on startup
- Written atomically (full file rewrite)
- ~200-300 bytes per track
- For 100K tracks: ~20-30MB

### 3. UMAP Coordinates (`umap_coords.json`)

**File**: `data/embeddings_cache/umap_coords.json`

```json
{
  "track_id_1": {
    "x": 12.345,
    "y": -6.789
  },
  "track_id_2": { ... }
}
```

**Characteristics**:
- 2D coordinates from UMAP projection
- Updated in batch after UMAP training
- ~50 bytes per track
- For 100K tracks: ~5MB

### 4. Index (`index.json`)

**File**: `data/embeddings_cache/index.json`

```json
{
  "track_id_1": "data/embeddings_cache/embedding_track_id_1.npy",
  "track_id_2": "data/embeddings_cache/embedding_track_id_2.npy"
}
```

**Characteristics**:
- Maps track IDs to embedding file paths
- Enables fast existence checks (`track_id in index`)
- Updated when new embeddings are added

## Data Flow

### Pipeline 1: Setup (One-time)

```
1. scrape_songs.py
   ↓
   Writes: base_songs.json
   {
     "total_tracks": 100000,
     "tracks": {
       "track_id": {
         "title": "...",
         "artist": "...",
         "source": "genre_diversity",
         "genre": "indie"
       }
     }
   }

2. generate_embeddings.py
   ↓
   Reads: base_songs.json
   Writes:
   - embedding_TRACK_ID.npy (one per track)
   - metadata.json (updated)
   - index.json (updated)

   Checkpoint every 1K songs:
   - Saves metadata.json
   - Saves index.json
   - Ensures fault tolerance

3. train_umap.py
   ↓
   Reads: All embedding_*.npy files (via index.json)
   Writes:
   - models/umap_model.pkl (fitted UMAP)
   - umap_coords.json (2D coordinates)
```

### Pipeline 2: User Visualization (Daily)

```
1. user_history.py
   ↓
   Fetches from Spotify API
   Returns: List of (track_id, timestamp)

2. embed_user_tracks.py
   ↓
   For each track_id:
   - Check index.json (exists?)
   - If yes: load embedding_TRACK_ID.npy
   - If no: generate embedding, save to disk
   ↓
   Load models/umap_model.pkl
   Transform embeddings to 2D
   Returns: {track_id: (x, y)}

3. visualize.py
   ↓
   Creates visualizations
   Saves to: data/visualizations/session_TIMESTAMP/
```

## Storage Operations

### Read Performance

| Operation | Time | Method |
|-----------|------|--------|
| Check if embedding exists | <1ms | `track_id in index` (dict lookup) |
| Load single embedding | 1-2ms | `np.load(file_path)` |
| Load 100 embeddings | 100-200ms | Sequential loading |
| Load metadata for track | <1ms | `metadata[track_id]` (dict lookup) |
| Load all embeddings | 5-10s | Load all .npy files into matrix |

### Write Performance

| Operation | Time | Method |
|-----------|------|--------|
| Save single embedding | 2-5ms | `np.save(file_path, embedding)` |
| Update metadata (1 track) | 50-100ms | Rewrite entire metadata.json |
| Batch update (1000 tracks) | 50-100ms | Single metadata.json write |
| Save index | 20-50ms | Rewrite entire index.json |

### Space Efficiency

For 100K tracks:
- Embeddings: 100K × 6KB = **~600MB**
- Metadata: **~30MB**
- UMAP coords: **~5MB**
- Index: **~10MB**
- **Total: ~650MB**

## Concurrency & Locking

### File-based Storage Advantages

✅ **Parallel reads**: Multiple processes can read different .npy files simultaneously

✅ **Isolated writes**: Each track writes to its own file (no conflicts)

✅ **Checkpoint safety**: Can save progress without blocking reads

### Limitations

⚠️ **JSON updates**: Metadata and index updates require full file rewrite
- Not atomic across multiple writers
- Should only update from single process

⚠️ **No built-in transactions**: Manual checkpoint management

## Upgrade Paths

### Option 1: PostgreSQL + pgvector

```sql
CREATE TABLE embeddings (
    track_id VARCHAR(50) PRIMARY KEY,
    embedding vector(768),
    umap_x FLOAT,
    umap_y FLOAT,
    metadata JSONB
);

CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
```

**Migration**:
```python
# Read current storage
db = EmbeddingDatabase()
track_ids, embeddings = db.get_all_embeddings()

# Write to PostgreSQL
for track_id, embedding in zip(track_ids, embeddings):
    metadata = db.get_metadata(track_id)
    coords = db.get_umap_coords(track_id)
    pg_db.insert(track_id, embedding, coords, metadata)
```

### Option 2: Qdrant (Vector Database)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(path="./qdrant_data")

client.create_collection(
    collection_name="music_embeddings",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# Insert embeddings
client.upsert(
    collection_name="music_embeddings",
    points=[
        {
            "id": track_id,
            "vector": embedding.tolist(),
            "payload": {
                "title": metadata["title"],
                "artist": metadata["artist"],
                "umap_x": coords[0],
                "umap_y": coords[1]
            }
        }
    ]
)
```

### Option 3: HDF5 (Single-file)

```python
import h5py

with h5py.File('embeddings.h5', 'w') as f:
    # Store all embeddings in single dataset
    f.create_dataset('embeddings', data=embeddings_matrix)
    f.create_dataset('track_ids', data=track_ids)

    # Store metadata as attributes
    for i, track_id in enumerate(track_ids):
        f['embeddings'].attrs[f'meta_{i}'] = json.dumps(metadata[track_id])
```

## Current Implementation: `database.py`

### Class: `EmbeddingDatabase`

**Initialization**:
```python
db = EmbeddingDatabase()  # Uses config.EMBEDDINGS_CACHE_DIR
```

**Key Methods**:

1. **Add embedding**:
```python
db.add_embedding(
    track_id="abc123",
    embedding=np.array([...]),  # 768-dim
    metadata={"title": "...", "artist": "..."},
    umap_coords=(12.3, -4.5)
)
```

2. **Get embedding**:
```python
embedding = db.get_embedding("abc123")  # Returns np.array or None
```

3. **Batch operations**:
```python
# Get multiple
embeddings = db.get_embeddings(["id1", "id2", "id3"])

# Add multiple
db.batch_add_embeddings([
    {"track_id": "id1", "embedding": emb1, "metadata": meta1},
    {"track_id": "id2", "embedding": emb2, "metadata": meta2}
])
```

4. **Get all embeddings** (for UMAP training):
```python
track_ids, embeddings_matrix = db.get_all_embeddings()
# Returns: (List[str], np.ndarray[N, 768])
```

5. **Save to disk**:
```python
db.save()  # Writes metadata.json, index.json, umap_coords.json
```

## Best Practices

### For Pipeline 1 (Setup)

✅ **Checkpoint frequently**:
```python
for i, track in enumerate(tracks):
    process_track(track)
    if i % 1000 == 0:
        db.save()  # Save progress
```

✅ **Batch updates**:
```python
# Don't save after each track
db.add_embedding(...)  # Adds to memory

# Save in batches
if len(batch) >= 1000:
    db.batch_add_embeddings(batch)  # Saves to disk once
```

### For Pipeline 2 (User)

✅ **Cache metadata in memory**:
```python
db = EmbeddingDatabase()  # Loads metadata.json into memory
metadata = db.get_metadata(track_id)  # Fast dict lookup
```

✅ **Lazy load UMAP model**:
```python
# Only load when needed
if self.umap_model is None:
    self.umap_model = joblib.load(config.UMAP_MODEL_PATH)
```

## Monitoring & Maintenance

### Check database stats:
```python
stats = db.get_stats()
print(stats)
# {
#   'total_tracks': 95000,
#   'tracks_with_metadata': 95000,
#   'tracks_with_umap_coords': 95000,
#   'cache_dir': 'data/embeddings_cache'
# }
```

### Cleanup:
```python
# Delete specific track
db.delete_track("bad_track_id")

# Clear everything (careful!)
db.clear_all()
```

### Backup:
```bash
# Simple file copy
tar -czf embeddings_backup.tar.gz data/embeddings_cache/

# Or rsync for incremental
rsync -av data/embeddings_cache/ backup/embeddings_cache/
```

## Summary

**Current System**:
- Simple file-based storage
- One .npy file per embedding
- JSON for metadata and coordinates
- Fast for <1M tracks
- Easy to understand and debug

**When to Upgrade**:
- **Vector DB**: Need similarity search or >1M tracks
- **PostgreSQL**: Want SQL queries + vector ops
- **HDF5**: Want single-file portability

**Key Design Principle**:
The `EmbeddingDatabase` class provides an abstraction that makes it easy to swap the underlying storage without changing the rest of the code.
