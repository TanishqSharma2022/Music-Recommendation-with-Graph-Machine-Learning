import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle 
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch.nn.modules.loss import _Loss
import random
from model import HeteroGNN  # Assuming you have a model defined in model.py
from utils import sample_negatives, sample_hard_negatives, recall_at_k, recall_at_k_fast  # Assuming these utility functions are
from torch_geometric.loader import LinkNeighborLoader
import torch.cuda.amp as amp  # Add this import

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)



# set the seed for reproducibility
seed = 224
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



# Note if you've already generated the graph above, you can skip those steps, and simply run set reload to True!
reload = True
if reload:
  sub_G = pickle.load(open("10K_playlist_graph.pkl", "rb"))
print('Num nodes:', sub_G.number_of_nodes(), '. Num edges:', sub_G.number_of_edges())




def nx_to_heterodata(G):
    data = HeteroData()

    # ---- 1. Map nodes per type ----
    node_maps = {}
    for node, attr in G.nodes(data=True):
        ntype = attr["node_type"]
        if ntype not in node_maps:
            node_maps[ntype] = {}
        node_maps[ntype][node] = len(node_maps[ntype])

    # ---- 2. Set num_nodes per node type ----
    for ntype, idmap in node_maps.items():
        data[ntype].num_nodes = len(idmap)

    # ---- 3. Collect edges grouped by (src_type, rel, dst_type) ----
    edge_groups = {}

    for u, v, attr in G.edges(data=True):
        rel = attr["edge_types"]               # your key
        src_t = G.nodes[u]["node_type"]
        dst_t = G.nodes[v]["node_type"]

        src_id = node_maps[src_t][u]
        dst_id = node_maps[dst_t][v]

        edge_type = (src_t, rel, dst_t)

        if edge_type not in edge_groups:
            edge_groups[edge_type] = []

        edge_groups[edge_type].append([src_id, dst_id])

    # ---- 4. Write to PyG ----
    for etype, edges in edge_groups.items():
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data[etype].edge_index = edge_index

    return data

hetero_data = nx_to_heterodata(sub_G)
print(hetero_data)

edge_types = hetero_data.edge_types


transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.0,
    split_labels=True,
    is_undirected=True,
    add_negative_train_samples=False,
    edge_types=[('playlist', 'track_in_playlist', 'track')],
    rev_edge_types=[('track', 'track_in_playlist', 'playlist')],
)


train_split, val_split, test_split = transform(hetero_data)


# Edge types
fwd_edge = ('playlist', 'track_in_playlist', 'track')
rev_edge = ('track', 'track_in_playlist', 'playlist')

def convert_edge_types(split, edge_type):
    # Convert message-passing edges
    if 'edge_index' in split[edge_type]:
        split[edge_type].edge_index = split[edge_type].edge_index.long()

    # Convert label edges (used for link prediction)
    if "edge_label_index" in split[edge_type]:
        split[edge_type].edge_label_index = split[edge_type].edge_label_index.long()

# Convert FORWARD + REVERSE edges
for split in [train_split, val_split, test_split]:
    convert_edge_types(split, fwd_edge)
    convert_edge_types(split, rev_edge)

# Print stats
print("Forward edge:", fwd_edge)
print("Reverse edge:", rev_edge)

print(f"Train supervision edges: {train_split[fwd_edge].pos_edge_label_index.shape[1]}")
print(f"Val supervision edges:   {val_split[fwd_edge].pos_edge_label_index.shape[1]}")
print(f"Test supervision edges:  {test_split[fwd_edge].pos_edge_label_index.shape[1]}")

print(f"Train MP edges (fwd):    {train_split[fwd_edge].edge_index.shape[1]}")
print(f"Train MP edges (rev):    {train_split[rev_edge].edge_index.shape[1]}")

print(f"Val MP edges (fwd):      {val_split[fwd_edge].edge_index.shape[1]}")
print(f"Val MP edges (rev):      {val_split[rev_edge].edge_index.shape[1]}")

print(f"Test MP edges (fwd):     {test_split[fwd_edge].edge_index.shape[1]}")
print(f"Test MP edges (rev):     {test_split[rev_edge].edge_index.shape[1]}")




def create_model_rgcn(metadata, num_nodes_dict, embedding_dim=128, num_layers=3):
        """Create RGCN model."""
        return HeteroGNN(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            model_type="RGCN"
        )

def create_model_sage(metadata, num_nodes_dict, embedding_dim=128, num_layers=3):
    """Create GraphSAGE-H model."""
    return HeteroGNN(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        model_type="SAGE"
    )

def create_model_hgt(metadata, num_nodes_dict, embedding_dim=128, num_layers=3, heads=4):
    """Create HGT model."""
    return HeteroGNN(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        model_type="HGT",
        heads=heads
    )





import torch

def edge_label_index_global_to_local(batch, edge_label_index, src_type, dst_type):
    """
    Convert edge_label_index that uses global node ids into local indices for the given batch.

    Args:
        batch: HeteroData batch from LinkNeighborLoader
        edge_label_index: LongTensor shape [2, num_edges] (global ids)
        src_type: source node type string, e.g. 'playlist'
        dst_type: destination node type string, e.g. 'track'

    Returns:
        local_edge_label_index: LongTensor shape [2, num_edges] (local indices in batch)
    """
    # If loader already relabeled, likely no `n_id` present; handle both cases
    if hasattr(batch[src_type], 'n_id') and hasattr(batch[dst_type], 'n_id'):
        src_nid = batch[src_type].n_id.cpu().numpy()
        dst_nid = batch[dst_type].n_id.cpu().numpy()

        # build maps (global_id -> local_idx)
        src_map = {int(g): i for i, g in enumerate(src_nid)}
        dst_map = {int(g): i for i, g in enumerate(dst_nid)}

        src_global = edge_label_index[0].cpu().numpy().astype(int)
        dst_global = edge_label_index[1].cpu().numpy().astype(int)

        # Convert; if any global id not found, mark as -1 to detect issues
        src_local = [src_map.get(g, -1) for g in src_global]
        dst_local = [dst_map.get(g, -1) for g in dst_global]

        src_local = torch.tensor(src_local, dtype=torch.long, device=edge_label_index.device)
        dst_local = torch.tensor(dst_local, dtype=torch.long, device=edge_label_index.device)

        local_idx = torch.vstack([src_local, dst_local])
        return local_idx
    else:
        # If n_id isn't available, assume edge_label_index is already local
        return edge_label_index.long().to(edge_label_index.device)









class HeteroTrainer:
    """
    Trainer for heterogeneous GNN link prediction with comprehensive logging and visualization.
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 optimizer, device, args):
        """
        Args:
            model: HeteroGNN model
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            optimizer: PyTorch optimizer
            device: Device to train on
            args: Dictionary with training arguments
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.args = args
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{args.get('model_name', 'model')}_{self.timestamp}"
        self.save_dir = os.path.join(args.get('output_dir', 'experiments'), self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'plots'), exist_ok=True)
        
        # Tracking metrics
        self.history = {
            'train_loss': [],
            'train_roc': [],
            'val_loss': [],
            'val_roc': [],
            'val_recall': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_roc = 0.0
        self.best_val_recall = 0.0
        
        # Save config
        self._save_config()
    
    def _save_config(self):
        """Save training configuration."""
        config = {
            'model_type': self.model.model_type,
            'embedding_dim': self.model.embedding_dim,
            'num_layers': self.model.num_layers,
            'args': self.args,
            'timestamp': self.timestamp
        }
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self, epoch, neg_sampling='random', hard_neg_every=5):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args["epochs"]} [Train]')
        
        scaler = amp.GradScaler()  # Initialize the gradient scaler for mixed precision

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            with amp.autocast():  # Enable mixed precision
                # Forward pass on positive edges
                pos_pred = self.model(batch)
                pos_label = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label
                
                # Sample negative edges
                if neg_sampling == 'random':
                    neg_edge_index, neg_edge_label = sample_negatives(
                        batch, 
                        neg_ratio=self.args.get('neg_ratio', 3)
                    )
                elif neg_sampling == 'hard':
                    # Use hard negatives periodically
                    if epoch % hard_neg_every == 0:
                        neg_edge_index, neg_edge_label = sample_hard_negatives(
                            batch, 
                            self.model,
                            device=self.device,
                            frac_sample=1.0 - (0.5 * epoch / self.args['epochs'])
                        )
                    else:
                        neg_edge_index, neg_edge_label = sample_negatives(batch, neg_ratio=3)
                
                # Forward pass on negative edges
                neg_pred = self.model(batch, neg_edge_index)
                
                # Compute loss
                loss_fn = self.args.get('loss_fn', 'BPR')
                if loss_fn == 'BPR':
                    loss = self.model.bpr_loss(pos_pred, neg_pred)
                elif loss_fn == 'BCE':
                    all_pred = torch.cat([pos_pred, neg_pred])
                    all_label = torch.cat([pos_label, neg_edge_label])
                    loss = self.model.link_pred_loss(all_pred, all_label)
            
            scaler.scale(loss).backward()  # Scale the loss and call backward
            
            # Gradient clipping
            if self.args.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            scaler.step(self.optimizer)  # Update the parameters
            scaler.update()  # Update the scale for next iteration
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions for ROC
            with torch.no_grad():
                all_preds.extend(torch.cat([pos_pred, neg_pred]).cpu().numpy())
                all_labels.extend(torch.cat([
                    torch.ones_like(pos_pred),
                    torch.zeros_like(neg_pred)
                ]).cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        train_roc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, train_roc
    
    @torch.no_grad()
    def validate(self, loader, desc='Val'):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        for batch in tqdm(loader, desc=f'[{desc}]', leave=False):
            batch = batch.to(self.device)
            batch['playlist','track_in_playlist','track'].pos_edge_label_index = (
            edge_label_index_global_to_local(
                batch,
                batch['playlist','track_in_playlist','track'].pos_edge_label_index,
                'playlist',
                'track'
            )
        )
            # Positive edges
            pos_pred = self.model(batch)
            pos_label = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label
            
            # Random negative edges
            neg_edge_index, neg_edge_label = sample_negatives(batch, neg_ratio=3)
            neg_pred = self.model(batch, neg_edge_index)
            
            # Compute loss
            loss_fn = self.args.get('loss_fn', 'BPR')
            if loss_fn == 'BPR':
                loss = self.model.bpr_loss(pos_pred, neg_pred)
            elif loss_fn == 'BCE':
                all_pred = torch.cat([pos_pred, neg_pred])
                all_label = torch.cat([pos_label, neg_edge_label])
                loss = self.model.link_pred_loss(all_pred, all_label)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions
            all_preds.extend(torch.cat([pos_pred, neg_pred]).cpu().numpy())
            all_labels.extend(torch.cat([
                torch.ones_like(pos_pred),
                torch.zeros_like(neg_pred)
            ]).cpu().numpy())
        
        avg_loss = total_loss / num_batches
        val_roc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, val_roc
    
    @torch.no_grad()
    def compute_recall_at_k(self, loader, k=10):
        """Compute Recall@K on loader."""
        self.model.eval()
        all_recalls = []
        
        for batch in tqdm(loader, desc=f'[Recall@{k}]', leave=False):
            batch = batch.to(self.device)
            recall = recall_at_k_fast(batch, self.model, k=k, device=self.device)
            all_recalls.append(recall)
        
        return np.mean(all_recalls)
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.exp_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.args['epochs']):
            import time
            epoch_start = time.time()
            
            # Train
            train_loss, train_roc = self.train_epoch(
                epoch, 
                neg_sampling=self.args.get('neg_sampling', 'random'),
                hard_neg_every=self.args.get('hard_neg_every', 5)
            )
            
            # Validate
            val_loss, val_roc = self.validate(self.val_loader, desc='Val')
            
            # Compute Recall@K periodically
            val_recall = None
            if epoch % self.args.get('recall_every', 5) == 0:
                val_recall = self.compute_recall_at_k(
                    self.val_loader, 
                    k=self.args.get('recall_k', 10)
                )
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_roc'].append(train_roc)
            self.history['val_loss'].append(val_loss)
            self.history['val_roc'].append(val_roc)
            if val_recall is not None:
                self.history['val_recall'].append(val_recall)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.args['epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train ROC: {train_roc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val ROC:   {val_roc:.4f}")
            if val_recall is not None:
                print(f"  Val Recall@{self.args.get('recall_k', 10)}: {val_recall:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_roc > self.best_val_roc:
                self.best_val_roc = val_roc
                self.save_checkpoint(epoch, 'best_roc')
                print(f"  → New best ROC: {val_roc:.4f}")
            
            if val_recall is not None and val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.save_checkpoint(epoch, 'best_recall')
                print(f"  → New best Recall: {val_recall:.4f}")
            
            # Save checkpoint periodically
            if epoch % self.args.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
            
            # Plot periodically
            if epoch % self.args.get('plot_every', 10) == 0:
                self.plot_metrics()
            
            # Learning rate scheduling
            if self.args.get('use_scheduler', False):
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(val_loss)
        
        # Final test evaluation
        print(f"\n{'='*60}")
        print("Training Complete! Running final test evaluation...")
        print(f"{'='*60}\n")
        
        test_loss, test_roc = self.validate(self.test_loader, desc='Test')
        test_recall = self.compute_recall_at_k(
            self.test_loader, 
            k=self.args.get('recall_k', 10)
        )
        
        print(f"\nFinal Test Results:")
        print(f"  Test Loss:   {test_loss:.4f}")
        print(f"  Test ROC:    {test_roc:.4f}")
        print(f"  Test Recall@{self.args.get('recall_k', 10)}: {test_recall:.4f}")
        
        # Save final results
        self.history['test_loss'] = test_loss
        self.history['test_roc'] = test_roc
        self.history['test_recall'] = test_recall
        
        self.save_history()
        self.plot_metrics()
        self.save_checkpoint(self.args['epochs']-1, 'final')
        
        print(f"\nAll results saved to: {self.save_dir}")
    
    def save_checkpoint(self, epoch, name):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_roc': self.best_val_roc,
            'best_val_recall': self.best_val_recall
        }
        path = os.path.join(self.save_dir, 'checkpoints', f'{name}.pt')
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history."""
        # Save as JSON
        history_path = os.path.join(self.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.save_dir, 'history.csv')
        df.to_csv(csv_path, index=False)
    
    def plot_metrics(self):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC plot
        axes[0, 1].plot(self.history['train_roc'], label='Train ROC', linewidth=2)
        axes[0, 1].plot(self.history['val_roc'], label='Val ROC', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROC AUC')
        axes[0, 1].set_title('Training and Validation ROC AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall plot
        if self.history['val_recall']:
            recall_epochs = list(range(0, len(self.history['val_recall']) * 
                                      self.args.get('recall_every', 5), 
                                      self.args.get('recall_every', 5)))
            axes[1, 0].plot(recall_epochs, self.history['val_recall'], 
                          'o-', linewidth=2, markersize=8, label=f'Val Recall@{self.args.get("recall_k", 10)}')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall@K')
            axes[1, 0].set_title(f'Validation Recall@{self.args.get("recall_k", 10)}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rate'], linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'training_metrics.png'), dpi=300)
        plt.close()



# Usage example
if __name__ == "__main__":
    # Training arguments
    args = {
        'model_name': 'HeteroGNN_RGCN',
        'epochs': 20,
        'loss_fn': 'BPR',  # 'BPR' or 'BCE'
        'neg_sampling': 'random',  # 'random' or 'hard'
        'neg_ratio': 30,
        'hard_neg_every': 1,
        'recall_every': 5,
        'recall_k': 30,
        'save_every': 10,
        'plot_every': 5,
        'clip_grad': True,
        'use_scheduler': False,
        'output_dir': 'experiments'
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model_rgcn(
        metadata=train_split.metadata(),
        num_nodes_dict={ntype: train_split[ntype].num_nodes for ntype in train_split.node_types},
        embedding_dim=128,
        num_layers=3
    )
    model = model.to(device)

    
    edge_label_index = train_split["playlist", "track_in_playlist", "track"].pos_edge_label_index
    edge_label = train_split["playlist", "track_in_playlist", "track"].pos_edge_label

    train_loader = LinkNeighborLoader(
        data=train_split,
        num_neighbors=[50, 50],  # Simpler: same for all edge types
        edge_label_index=(("playlist", "track_in_playlist", "track"), edge_label_index),
        edge_label=edge_label,
        batch_size=512,
        shuffle=True,
        num_workers=12  # Add this line to improve data loading
    )

    # Validation Loader
    val_edge_label_index = val_split["playlist", "track_in_playlist", "track"].pos_edge_label_index
    val_edge_label = val_split["playlist", "track_in_playlist", "track"].pos_edge_label

    val_loader = LinkNeighborLoader(
        data=val_split,
        num_neighbors=[20, 10],
        edge_label_index=(("playlist", "track_in_playlist", "track"), val_edge_label_index),
        edge_label=val_edge_label,
        batch_size=512,
        shuffle=False,  # No shuffle for evaluation
        num_workers=4  # Add this line to improve data loading
    )

    # Test Loader
    test_edge_label_index = test_split["playlist", "track_in_playlist", "track"].pos_edge_label_index
    test_edge_label = test_split["playlist", "track_in_playlist", "track"].pos_edge_label

    test_loader = LinkNeighborLoader(
        data=test_split,
        num_neighbors=[20, 10],
        edge_label_index=(("playlist", "track_in_playlist", "track"), test_edge_label_index),
        edge_label=test_edge_label,
        batch_size=512,
        shuffle=False,  # No shuffle for evaluation
        num_workers=4  # Add this line to improve data loading
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # print("="*60)
    # print("SANITY CHECK: Single Batch Test")
    # print("="*60)
    
    # # Test TRAIN loader
    # print("\n[TRAIN LOADER]")
    # for batch in train_loader:
    #     batch = batch.to(device)
        
    #     # Forward pass
    #     pos_pred = model(batch)
    #     print(f"✓ Forward pass works: {pos_pred.shape}")
        
    #     # Loss computation
    #     from utils import sample_negatives
    #     neg_edge_index, neg_edge_label = sample_negatives(batch, device=device, neg_ratio=3)
    #     neg_pred = model(batch, neg_edge_index)
    #     loss = model.bpr_loss(pos_pred, neg_pred)
    #     print(f"✓ Loss computation works: {loss.item():.4f}")
        
    #     # Backward pass
    #     loss.backward()
    #     print(f"✓ Backward pass works")
    #     break
    
    # # Test VAL loader
    # print("\n[VAL LOADER]")
    # for batch in val_loader:
    #     batch = batch.to(device)
    #     batch['playlist','track_in_playlist','track'].pos_edge_label_index = (
    #         edge_label_index_global_to_local(
    #             batch,
    #             batch['playlist','track_in_playlist','track'].pos_edge_label_index,
    #             'playlist',
    #             'track'
    #         )
    #     )

    #     # Forward pass
    #     pos_pred = model(batch)
    #     print(f"✓ Forward pass works: {pos_pred.shape}")
        
    #     # Loss computation
    #     neg_edge_index, neg_edge_label = sample_negatives(batch, device=device, neg_ratio=3)
    #     neg_pred = model(batch, neg_edge_index)
    #     loss = model.bpr_loss(pos_pred, neg_pred)
    #     print(f"✓ Loss computation works: {loss.item():.4f}")
    #     print(f"✓ Validation batch OK")
    #     break
    
    # # Test TEST loader
    # print("\n[TEST LOADER]")
    # for batch in test_loader:
    #     batch = batch.to(device)
    #     batch['playlist','track_in_playlist','track'].pos_edge_label_index = (
    #         edge_label_index_global_to_local(
    #             batch,
    #             batch['playlist','track_in_playlist','track'].pos_edge_label_index,
    #             'playlist',
    #             'track'
    #         )
    #     )
    #     # Forward pass
    #     pos_pred = model(batch)
    #     print(f"✓ Forward pass works: {pos_pred.shape}")
        
    #     # Loss computation
    #     neg_edge_index, neg_edge_label = sample_negatives(batch, device=device, neg_ratio=3)
    #     neg_pred = model(batch, neg_edge_index)
    #     loss = model.bpr_loss(pos_pred, neg_pred)
    #     print(f"✓ Loss computation works: {loss.item():.4f}")
    #     print(f"✓ Test batch OK")
    #     break
    
    # print("\n" + "="*60)
    # print("✓ ALL SANITY CHECKS PASSED!")
    # print("="*60 + "\n")
    
    # Initialize trainer
    trainer = HeteroTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        args=args
    )
    
    # Train
    trainer.train()