# Music Recommendation with Graph Neural Networks  
### *Heterogeneous GraphSAGE / RGCN / HGT for Playlist-Track Prediction using Spotify Million Playlist Dataset*

---

## Introduction

This project builds a **Heterogeneous Graph Neural Network (HGNN)** for **playlist continuation** and **music recommendation** using the **Spotify Million Playlist Dataset (MPD)**.

Unlike classical collaborative filtering, the model uses:

- **Playlist → Track edges**
- **Track → Artist edges**
- **Track → Album edges**

This richer graph structure allows better modeling of:

- music similarity  
- playlist semantics  
- cold-start behaviour  
- artist / album influence on track embeddings  

The goal:  
**Predict which tracks should appear in a given playlist (link prediction).**

---

## Implementation 

- Heterogeneous Graph with 4 node types  
- PyTorch Geometric based training pipeline  
- LinkNeighborLoader mini-batching  
- Negative sampling (Random + Hard negatives)  
- BCE / BPR loss options  
- Multiple models: GraphSAGE-H, RGCN, HGT  
- Complete evaluation suite (ROC, Recall@K)  
- Track audio feature extraction using Spotify API  
- Reproducible training with detailed logging  


---

## Dataset

### **Spotify Million Playlist Dataset (MPD)**  
- 1,000,000 playlists  
- ~2M tracks  
- Track metadata  
- Artist, Album information  

This project uses a **50-file subset (~10k playlists)** for demonstration.

### Node Types
| Node | Description |
|------|-------------|
| Playlist | A single playlist from MPD |
| Track | Track URI, name |
| Artist | Artist URI |
| Album | Album URI |

### Edge Types
| Relation | Meaning |
|---------|----------|
| playlist → track | Track appears in playlist |
| track → artist | Track created by artist |
| track → album | Track belongs to album |

---

## Working 

This is the **full ML pipeline** used in the project.

### **1. Load & Parse Raw JSON Files**
Using `plot_actual_graph.py`, we extract:
- playlists  
- tracks  
- albums  
- artists  

### **2. Build a NetworkX Graph**
Nodes = playlists + tracks + albums + artists  
Edges = playlist-track + track-artist + track-album

### **3. Convert to PyTorch Geometric HeteroData**
Steps:
- integer node ID mapping  
- edge grouping by type  
- adjacency formatting  
- tensor conversion  

### **4. Apply Train/Val/Test Link Split**
Using RandomLinkSplit:

- 80% train  
- 10% validation  
- 10% test  

### **5. Mini-batch Sampling with LinkNeighborLoader**
Example:

```python
train_loader = LinkNeighborLoader(
    data=train_split,
    num_neighbors=[20, 10],
    batch_size=512,
    edge_label_index=train_edge_index,
    edge_label=train_edge_label
)
````

### **6. Train HGNN Models**

Supported models:

* **RGCN**
* **GraphSAGE-H**
* **HGT (Heterogeneous Graph Transformer)**

### **7. Link Prediction Head**

Prediction = dot product between playlist & track embeddings.

### **8. Loss Functions**

* BCEWithLogitsLoss (stable)
* BPR Loss (ranking-based)

### **9. Evaluation Metrics**

* ROC AUC
* Recall@K
* Training loss curves
* Gradient norms

---

## Architecture

### **Hetero GraphSAGE**

* Separate message passing for each edge type
* Combine all messages with weighted averaging

### **RGCN**

* Relation-specific transformation matrices



## Audio Feature Extraction

The script `fetch_audio_features.py`:

* Generates Spotify access tokens
* Fetches audio features for 100 tracks/batch
* Resumes automatically from checkpoints
* Handles rate limiting + failed requests

Example:

```bash
python fetch_audio_features.py
```


→ Added relation weights:

```
playlist-track: 1.0  
track-album:   0.3  
track-artist:  0.2  
```

### Final Results

* ROC AUC: **0.73**
* Recall@10: **0.004+** (from near 0 initially)

---


### Embeddings

Tracks cluster meaningfully by:

* genre
* album
* artist
* playlist co-occurrence

### Model Performance

| Metric    | Before Fixes | After Fixes |
| --------- | ------------ | ----------- |
| ROC AUC   | 0.60         | **0.73**    |
| Recall@10 | 0.00001      | **0.004+**  |

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torch_geometric networkx tqdm matplotlib requests
```

### 2. Download Spotify MPD

Place JSON files in:

```
spotify_million_playlist_dataset/data
```

### 3. Run Graph Building

```bash
python plot_actual_graph.py
```

### 4. Train Model

Use `core.ipynb`:

* load graph
* convert to PyG
* train GraphSAGE-H / RGCN / HGT
* evaluate



## Acknowledgements

* Spotify Research MPD Dataset
* CS224W Playlist GNN tutorial
* PyTorch Geometric team




