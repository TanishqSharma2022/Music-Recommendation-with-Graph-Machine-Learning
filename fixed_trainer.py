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
from model import HeteroGNN
from utils import sample_negatives, sample_hard_negatives, recall_at_k, recall_at_k_fast
from torch_geometric.loader import LinkNeighborLoader
import torch.cuda.amp as amp
import time

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)


# Enable TF32 for A100 (3x speedup with minimal accuracy loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# set the seed for reproducibility
seed = 224
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



# Note if you've already generated the graph above, you can skip those steps, and simply run set reload to True!
reload = True
if reload:
  sub_G = pickle.load(open("1K_playlist_graph.pkl", "rb"))
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
        
        # Enhanced tracking metrics - capture EVERY epoch
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_roc': [],
            'train_batch_count': [],
            'val_loss': [],
            'val_roc': [],
            'val_recall': [],
            'val_recall_epoch': [],  # Track which epochs have recall computed
            'learning_rate': [],
            'epoch_time': [],
            'gradient_norm': [],
            'best_val_roc_so_far': [],
            'best_val_recall_so_far': []
        }
        
        self.best_val_roc = 0.0
        self.best_val_roc_epoch = -1
        self.best_val_recall = 0.0
        self.best_val_recall_epoch = -1
        
        # Save config
        self._save_config()
        self._print_metrics_summary()
    
    def _print_metrics_summary(self):
        """Print summary of metrics that will be tracked."""
        print(f"\n{'='*70}")
        print("METRICS THAT WILL BE TRACKED AND VISUALIZED:")
        print(f"{'='*70}")
        print("\n📊 PER-EPOCH METRICS (tracked every epoch):")
        print("  • Train Loss (BPR/BCE loss)")
        print("  • Train ROC AUC (on sampled pos/neg edges)")
        print("  • Validation Loss (on validation edges)")
        print("  • Validation ROC AUC")
        print("  • Learning Rate (if scheduler enabled)")
        print("  • Epoch Time (training time)")
        print("  • Gradient Norm (training stability)")
        print("  • Best Val ROC & Recall Progress")
        
        recall_every = self.args.get('recall_every', 5)
        print(f"\n🎯 PERIODIC METRICS (tracked every {recall_every} epochs):")
        print(f"  • Validation Recall@{self.args.get('recall_k', 10)}")
        
        print(f"\n✅ FINAL METRICS (after training completes):")
        print("  • Test Loss")
        print("  • Test ROC AUC")
        print(f"  • Test Recall@{self.args.get('recall_k', 10)}")
        print(f"{'='*70}\n")
    
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
        total_grad_norm = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args["epochs"]} [Train]')
        
        scaler = amp.GradScaler()

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            with amp.autocast():
                # pos_pred = self.model(batch)
                pos_edge_index = batch['playlist','track_in_playlist','track'].edge_label_index
                pos_edge_index = edge_label_index_global_to_local(
                    batch,
                    pos_edge_index,
                    'playlist',
                    'track'
                )

                pos_pred = self.model(batch, pos_edge_index)

                pos_label = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label
                
                if neg_sampling == 'random':
                    neg_edge_index, neg_edge_label = sample_negatives(
                        batch, 
                        neg_ratio=self.args.get('neg_ratio', 3)
                    )
                elif neg_sampling == 'hard':
                    if epoch % hard_neg_every == 0:
                        neg_edge_index, neg_edge_label = sample_hard_negatives(
                            batch, 
                            self.model,
                            device=self.device,
                            frac_sample=1.0 - (0.5 * epoch / self.args['epochs'])
                        )
                    else:
                        neg_edge_index, neg_edge_label = sample_negatives(batch, neg_ratio=3)
                
                neg_edge_index = edge_label_index_global_to_local(
                    batch,
                    neg_edge_index,
                    'playlist',
                    'track'
                )
                neg_pred = self.model(batch, neg_edge_index)

                
                loss_fn = self.args.get('loss_fn', 'BPR')
                if loss_fn == 'BPR':
                    loss = self.model.bpr_loss(pos_pred, neg_pred)
                elif loss_fn == 'BCE':
                    all_pred = torch.cat([pos_pred, neg_pred])
                    all_label = torch.cat([pos_label, neg_edge_label])
                    loss = self.model.link_pred_loss(all_pred, all_label)
            
            scaler.scale(loss).backward()
            
            # Track gradient norm for training stability
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm
            
            if self.args.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            with torch.no_grad():
                all_preds.extend(torch.cat([pos_pred, neg_pred]).cpu().numpy())
                all_labels.extend(torch.cat([
                    torch.ones_like(pos_pred),
                    torch.zeros_like(neg_pred)
                ]).cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'grad_norm': f'{grad_norm:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches
        train_roc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, train_roc, avg_grad_norm, num_batches
    
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
            pos_edge_index = batch['playlist','track_in_playlist','track'].pos_edge_label_index
            pos_pred = self.model(batch, pos_edge_index)

            pos_label = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label
            
            neg_edge_index, neg_edge_label = sample_negatives(batch, neg_ratio=3)
            neg_pred = self.model(batch, neg_edge_index)
            
            loss_fn = self.args.get('loss_fn', 'BPR')
            if loss_fn == 'BPR':
                loss = self.model.bpr_loss(pos_pred, neg_pred)
            elif loss_fn == 'BCE':
                all_pred = torch.cat([pos_pred, neg_pred])
                all_label = torch.cat([pos_label, neg_edge_label])
                loss = self.model.link_pred_loss(all_pred, all_label)
            
            total_loss += loss.item()
            num_batches += 1
            
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
        print(f"\n{'='*70}")
        print(f"🚀 Starting training: {self.exp_name}")
        print(f"{'='*70}\n")
        
        training_start_time = time.time()
        
        for epoch in range(self.args['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_roc, avg_grad_norm, num_batches = self.train_epoch(
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
            
            # Track ALL metrics every epoch
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_roc'].append(train_roc)
            self.history['train_batch_count'].append(num_batches)
            self.history['val_loss'].append(val_loss)
            self.history['val_roc'].append(val_roc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            self.history['gradient_norm'].append(avg_grad_norm)
            
            # Track best values
            self.history['best_val_roc_so_far'].append(self.best_val_roc)
            self.history['best_val_recall_so_far'].append(self.best_val_recall)
            
            if val_recall is not None:
                self.history['val_recall'].append(val_recall)
                self.history['val_recall_epoch'].append(epoch + 1)
            
            # Print epoch summary
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{self.args['epochs']} Summary:")
            print(f"{'─'*70}")
            print(f"  📈 Train:  Loss={train_loss:.4f} | ROC={train_roc:.4f}")
            print(f"  📊 Val:    Loss={val_loss:.4f} | ROC={val_roc:.4f}")
            if val_recall is not None:
                print(f"  🎯 Recall@{self.args.get('recall_k', 10)}: {val_recall:.4f}")
            print(f"  ⚡ Grad Norm: {avg_grad_norm:.4f}")
            print(f"  ⏱️  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_roc > self.best_val_roc:
                self.best_val_roc = val_roc
                self.best_val_roc_epoch = epoch + 1
                self.save_checkpoint(epoch, 'best_roc')
                print(f"  ✨ New best ROC: {val_roc:.4f}")
            
            if val_recall is not None and val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.best_val_recall_epoch = epoch + 1
                self.save_checkpoint(epoch, 'best_recall')
                print(f"  ✨ New best Recall: {val_recall:.4f}")
            
            # Save checkpoint periodically
            if epoch % self.args.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
            
            # Plot periodically
            if epoch % self.args.get('plot_every', 5) == 0:
                self.plot_metrics()
            
            # Learning rate scheduling
            if self.args.get('use_scheduler', False):
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(val_loss)
        
        total_training_time = time.time() - training_start_time
        
        # Final test evaluation
        print(f"\n{'='*70}")
        print("🎯 Training Complete! Running final test evaluation...")
        print(f"{'='*70}\n")
        
        test_loss, test_roc = self.validate(self.test_loader, desc='Test')
        test_recall = self.compute_recall_at_k(
            self.test_loader, 
            k=self.args.get('recall_k', 10)
        )
        
        print(f"\n{'─'*70}")
        print(f"📋 FINAL TEST RESULTS:")
        print(f"{'─'*70}")
        print(f"  Test Loss:   {test_loss:.4f}")
        print(f"  Test ROC:    {test_roc:.4f}")
        print(f"  Test Recall@{self.args.get('recall_k', 10)}: {test_recall:.4f}")
        print(f"  Total Training Time: {total_training_time/60:.2f} minutes")
        
        # Save final results
        self.history['test_loss'] = test_loss
        self.history['test_roc'] = test_roc
        self.history['test_recall'] = test_recall
        self.history['total_training_time'] = total_training_time
        
        self.save_history()
        self.plot_metrics()
        self._save_summary()
        self.save_checkpoint(self.args['epochs']-1, 'final')
        
        print(f"\n{'='*70}")
        print(f"✅ All results saved to: {self.save_dir}")
        print(f"{'='*70}\n")
    
    def _save_summary(self):
        """Save training summary with best models and final metrics."""
        summary = {
            'experiment_name': self.exp_name,
            'timestamp': self.timestamp,
            'total_epochs': self.args['epochs'],
            'total_training_time_seconds': self.history['total_training_time'],
            'total_training_time_minutes': self.history['total_training_time'] / 60,
            'best_validation_roc': {
                'value': float(self.best_val_roc),
                'epoch': self.best_val_roc_epoch
            },
            'best_validation_recall': {
                'value': float(self.best_val_recall),
                'epoch': self.best_val_recall_epoch
            },
            'final_test_metrics': {
                'loss': float(self.history['test_loss']),
                'roc_auc': float(self.history['test_roc']),
                'recall_at_k': float(self.history['test_recall']),
                'recall_k_value': self.args.get('recall_k', 10)
            },
            'training_config': self.args
        }
        
        summary_path = os.path.join(self.save_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
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
        """Save training history to JSON and CSV."""
        # Save as JSON
        history_path = os.path.join(self.save_dir, 'history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, val in self.history.items():
            if isinstance(val, list):
                history_json[key] = [float(v) if isinstance(v, (np.number, torch.Tensor)) else v for v in val]
            else:
                history_json[key] = val
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        # Save as CSV
        csv_path = os.path.join(self.save_dir, 'history.csv')
        
        # Create DataFrame from history, handling different lengths
        df_dict = {}
        max_len = max(len(v) for v in self.history.values() if isinstance(v, list))
        
        for key, val in self.history.items():
            if isinstance(val, list):
                # Pad with NaN if needed
                padded = val + [np.nan] * (max_len - len(val))
                df_dict[key] = padded
        
        df = pd.DataFrame(df_dict)
        df.to_csv(csv_path, index=False)
        print(f"  📁 Saved history to {history_path} and {csv_path}")
    
    def plot_metrics(self):
        """Create comprehensive visualizations of all metrics."""
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Loss curves (top left)
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.history['epoch'], self.history['train_loss'], 'o-', label='Train', linewidth=2)
        ax1.plot(self.history['epoch'], self.history['val_loss'], 's-', label='Val', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.set_title('Loss Over Time', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC AUC curves (top middle)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(self.history['epoch'], self.history['train_roc'], 'o-', label='Train', linewidth=2)
        ax2.plot(self.history['epoch'], self.history['val_roc'], 's-', label='Val', linewidth=2)
        ax2.axhline(y=self.best_val_roc, color='r', linestyle='--', alpha=0.5, label=f'Best Val: {self.best_val_roc:.4f}')
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('ROC AUC', fontsize=10)
        ax2.set_title('ROC AUC Over Time', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        # 3. Recall@K (top right) - only plot where available
        ax3 = plt.subplot(3, 3, 3)
        if len(self.history['val_recall']) > 0:
            ax3.plot(self.history['val_recall_epoch'], self.history['val_recall'], 'D-', 
                    color='green', linewidth=2, markersize=8, label='Recall')
            ax3.axhline(y=self.best_val_recall, color='r', linestyle='--', alpha=0.5, 
                        label=f'Best: {self.best_val_recall:.4f}')
            ax3.set_xlabel('Epoch', fontsize=10)
            ax3.set_ylabel(f'Recall@{self.args.get("recall_k", 10)}', fontsize=10)
            ax3.set_title(f'Recall@{self.args.get("recall_k", 10)} Over Time', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Recall data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Recall@K (Not computed)', fontsize=12, fontweight='bold')
        
        # 4. Learning Rate (middle left)
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.history['epoch'], self.history['learning_rate'], 'o-', color='purple', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('Learning Rate', fontsize=10)
        ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Gradient Norm (middle center)
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(self.history['epoch'], self.history['gradient_norm'], 'o-', color='orange', linewidth=2)
        ax5.set_xlabel('Epoch', fontsize=10)
        ax5.set_ylabel('Gradient Norm', fontsize=10)
        ax5.set_title('Training Gradient Norm', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Epoch Time (middle right)
        ax6 = plt.subplot(3, 3, 6)
        ax6.bar(self.history['epoch'], self.history['epoch_time'], color='steelblue', alpha=0.7)
        ax6.set_xlabel('Epoch', fontsize=10)
        ax6.set_ylabel('Time (seconds)', fontsize=10)
        ax6.set_title('Epoch Training Time', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Best Val ROC Progress (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(self.history['epoch'], self.history['best_val_roc_so_far'], 'o-', 
                color='darkgreen', linewidth=2, markersize=6)
        ax7.set_xlabel('Epoch', fontsize=10)
        ax7.set_ylabel('Best Val ROC (so far)', fontsize=10)
        ax7.set_title('Best Validation ROC Progress', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1.05])
        
        # 8. Best Val Recall Progress (bottom middle)
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(self.history['epoch'], self.history['best_val_recall_so_far'], 'o-', 
                color='darkred', linewidth=2, markersize=6)
        ax8.set_xlabel('Epoch', fontsize=10)
        ax8.set_ylabel('Best Val Recall (so far)', fontsize=10)
        ax8.set_title('Best Validation Recall Progress', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9. Batch Count per Epoch (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        ax9.bar(self.history['epoch'], self.history['train_batch_count'], color='coral', alpha=0.7)
        ax9.set_xlabel('Epoch', fontsize=10)
        ax9.set_ylabel('Number of Batches', fontsize=10)
        ax9.set_title('Training Batches per Epoch', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Training Metrics: {self.exp_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.save_dir, 'plots', 'all_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved plot to {plot_path}")
        plt.close()
        
        # Create additional detailed plots
        self._plot_loss_detail()
        self._plot_roc_detail()
        if len(self.history['val_recall']) > 0:
            self._plot_recall_detail()
    
    def _plot_loss_detail(self):
        """Detailed loss visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear scale
        axes[0].plot(self.history['epoch'], self.history['train_loss'], 'o-', label='Train', linewidth=2)
        axes[0].plot(self.history['epoch'], self.history['val_loss'], 's-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Loss - Linear Scale', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Log scale (if loss values are positive)
        if min(self.history['train_loss'] + self.history['val_loss']) > 0:
            axes[1].semilogy(self.history['epoch'], self.history['train_loss'], 'o-', label='Train', linewidth=2)
            axes[1].semilogy(self.history['epoch'], self.history['val_loss'], 's-', label='Val', linewidth=2)
            axes[1].set_title('Loss - Log Scale', fontsize=12, fontweight='bold')
        else:
            axes[1].plot(self.history['epoch'], self.history['train_loss'], 'o-', label='Train', linewidth=2)
            axes[1].plot(self.history['epoch'], self.history['val_loss'], 's-', label='Val', linewidth=2)
            axes[1].set_title('Loss (Log scale not applicable)', fontsize=12, fontweight='bold')
        
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Loss Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'plots', 'loss_detail.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved detailed loss plot to {plot_path}")
        plt.close()
    
    def _plot_roc_detail(self):
        """Detailed ROC AUC visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC trends
        axes[0].plot(self.history['epoch'], self.history['train_roc'], 'o-', label='Train', linewidth=2.5, markersize=6)
        axes[0].plot(self.history['epoch'], self.history['val_roc'], 's-', label='Val', linewidth=2.5, markersize=6)
        axes[0].axhline(y=self.best_val_roc, color='r', linestyle='--', alpha=0.6, linewidth=2, 
                       label=f'Best Val: {self.best_val_roc:.4f} (Epoch {self.best_val_roc_epoch})')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('ROC AUC', fontsize=11)
        axes[0].set_title('ROC AUC Trends', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # ROC gap (train - val)
        roc_gap = [t - v for t, v in zip(self.history['train_roc'], self.history['val_roc'])]
        axes[1].plot(self.history['epoch'], roc_gap, 'D-', color='darkred', linewidth=2, markersize=6)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(self.history['epoch'], roc_gap, 0, where=[g >= 0 for g in roc_gap], 
                            alpha=0.3, color='red', label='Overfitting')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Train ROC - Val ROC', fontsize=11)
        axes[1].set_title('Generalization Gap', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('ROC AUC Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'plots', 'roc_detail.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved detailed ROC plot to {plot_path}")
        plt.close()
    
    def _plot_recall_detail(self):
        """Detailed Recall@K visualization."""
        if len(self.history['val_recall']) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Recall trend
        axes[0].plot(self.history['val_recall_epoch'], self.history['val_recall'], 'D-', 
                    color='green', linewidth=2.5, markersize=8, label='Recall')
        axes[0].axhline(y=self.best_val_recall, color='r', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'Best: {self.best_val_recall:.4f} (Epoch {self.best_val_recall_epoch})')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel(f'Recall@{self.args.get("recall_k", 10)}', fontsize=11)
        axes[0].set_title('Recall@K Trend', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Recall improvement per epoch
        recall_improvement = [0] + [self.history['val_recall'][i] - self.history['val_recall'][i-1] 
                                    for i in range(1, len(self.history['val_recall']))]
        axes[1].bar(self.history['val_recall_epoch'], recall_improvement, 
                   color=['green' if x > 0 else 'red' for x in recall_improvement], alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Recall Improvement', fontsize=11)
        axes[1].set_title('Epoch-to-Epoch Recall Change', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Recall@K Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'plots', 'recall_detail.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved detailed recall plot to {plot_path}")
        plt.close()


# ============================================================================
# TRAINING SCRIPT - SAGE MODEL
# ============================================================================

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"🖥️  Using device: {device}")
    print(f"{'='*70}\n")
    
    # Model configuration
    metadata = hetero_data.metadata()
    num_nodes_dict = {node_type: hetero_data[node_type].num_nodes for node_type in hetero_data.node_types}
    
    print(f"Metadata: {metadata}")
    print(f"Num nodes dict: {num_nodes_dict}\n")
    
    # Create SAGE model
    print("Creating SAGE model...")
    model = create_model_sage(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        embedding_dim=128,
        num_layers=3
    )
    model = model.to(device)
    print(f"Model created: {model.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create data loaders
    print("Creating data loaders...")
    
    # Training Loader
    train_edge_label_index = train_split["playlist", "track_in_playlist", "track"].pos_edge_label_index
    train_edge_label = train_split["playlist", "track_in_playlist", "track"].pos_edge_label

    train_loader = LinkNeighborLoader(
        data=train_split,
        num_neighbors=[15, 10],
        edge_label_index=(("playlist", "track_in_playlist", "track"), train_edge_label_index),
        edge_label=train_edge_label,
        batch_size=512,
        shuffle=True,
        num_workers=4
    )

    # Validation Loader
    val_edge_label_index = val_split["playlist", "track_in_playlist", "track"].pos_edge_label_index
    val_edge_label = val_split["playlist", "track_in_playlist", "track"].pos_edge_label

    val_loader = LinkNeighborLoader(
        data=val_split,
        num_neighbors=[15, 10],
        edge_label_index=(("playlist", "track_in_playlist", "track"), val_edge_label_index),
        edge_label=val_edge_label,
        batch_size=512,
        shuffle=False,
        num_workers=4
    )

    # Test Loader
    test_edge_label_index = test_split["playlist", "track_in_playlist", "track"].pos_edge_label_index
    test_edge_label = test_split["playlist", "track_in_playlist", "track"].pos_edge_label

    test_loader = LinkNeighborLoader(
        data=test_split,
        num_neighbors=[15, 10],
        edge_label_index=(("playlist", "track_in_playlist", "track"), test_edge_label_index),
        edge_label=test_edge_label,
        batch_size=512,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training arguments
    args = {
        'model_name': 'SAGE_A100',
        'output_dir': 'experiments',
        'epochs': 30,              # ← INCREASE from 15 (now we can afford it)
        'neg_ratio': 2,            # ← Slightly increase (was 1, now 2)
        'loss_fn': 'BPR',
        'neg_sampling': 'random',
        'hard_neg_every': 10,
        'recall_k': 10,
        'recall_every': 5,         # ← Can afford to compute more often
        'save_every': 5,
        'plot_every': 5,
        'clip_grad': True,
        'use_scheduler': False
    }
    
    # Initialize trainer
    print(f"\n{'='*70}")
    print("Initializing trainer...")
    print(f"{'='*70}\n")
    
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
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    trainer.train()
    
    print(f"\n{'='*70}")
    print("✅ Training completed successfully!")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"{'='*70}\n")