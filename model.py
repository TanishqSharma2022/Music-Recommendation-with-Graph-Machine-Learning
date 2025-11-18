import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, ModuleList, Linear
from torch_geometric.nn import RGCNConv, SAGEConv, HGTConv, to_hetero
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, Union, Dict, List
import torch.cuda.amp as amp  # Add this import


class HeteroGNN(torch.nn.Module):
    """
    Base heterogeneous GNN model for playlist-track recommendation.
    Supports RGCN, GraphSAGE-H, and HGT architectures.
    """

    def __init__(
        self,
        metadata: tuple,  # (node_types, edge_types)
        num_nodes_dict: Dict[str, int],  # {'playlist': 10000, 'track': 171855, ...}
        embedding_dim: int,
        num_layers: int,
        model_type: str = "RGCN",  # "RGCN", "SAGE", "HGT"
        alpha: Optional[Union[float, Tensor]] = None,
        heads: int = 4,  # For HGT
        **kwargs,
    ):
        super().__init__()
        
        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Alpha weighting for layer combinations
        if alpha is None:
            alpha = 1. / (num_layers + 1)
        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)
        
        # Create embeddings for each node type
        self.embeddings = nn.ModuleDict({
            node_type: Embedding(num_nodes, embedding_dim)
            for node_type, num_nodes in num_nodes_dict.items()
        })
        
        # Create convolutional layers based on model type
        self.convs = ModuleList()
        
        if model_type == "RGCN":
            # Relational GCN - handles different edge types with relation-specific weights
            num_relations = len(self.edge_types)
            for _ in range(num_layers):
                self.convs.append(
                    RGCNConv(
                        embedding_dim, 
                        embedding_dim, 
                        num_relations=num_relations,
                        **kwargs
                    )
                )
        
        elif model_type == "SAGE":
            # GraphSAGE for heterogeneous graphs (will be wrapped with to_hetero)
            for _ in range(num_layers):
                self.convs.append(
                    SAGEConv(embedding_dim, embedding_dim, **kwargs)
                )
        
        elif model_type == "HGT":
            # Heterogeneous Graph Transformer
            for _ in range(num_layers):
                self.convs.append(
                    HGTConv(
                        embedding_dim,
                        embedding_dim,
                        metadata,
                        heads=heads,
                        **kwargs
                    )
                )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embeddings.values():
            torch.nn.init.xavier_uniform_(emb.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, batch) -> Dict[str, Tensor]:
        """
        Get node embeddings for heterogeneous batch.
        
        Args:
            batch: HeteroData batch
            
        Returns:
            Dictionary of embeddings for each node type
        """
        # Initialize with learned embeddings using node_id (global IDs in batch)
        # This indexes the FULL embedding table with batch's global node IDs
        x_dict = {
            node_type: self.embeddings[node_type](batch[node_type].n_id)
            for node_type in self.node_types
        }
        
        weights = self.alpha.softmax(dim=-1)
        
        # Initialize output with weighted initial embeddings
        out_dict = {
            node_type: x * weights[0]
            for node_type, x in x_dict.items()
        }
        
        # Message passing
        if self.model_type == "RGCN":
            # RGCN uses homogeneous edge_index with edge_type
            edge_index, edge_type = self._to_homogeneous(batch)
            
            # Convert node features to homogeneous
            x = torch.cat([x_dict[nt] for nt in self.node_types], dim=0)
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index, edge_type)
                x = x.relu()
                
                # Add to output with weight
                start_idx = 0
                for node_type in self.node_types:
                    num_nodes = x_dict[node_type].size(0)
                    out_dict[node_type] = out_dict[node_type] + x[start_idx:start_idx+num_nodes] * weights[i + 1]
                    start_idx += num_nodes
        
        elif self.model_type == "SAGE":
            # GraphSAGE with heterogeneous message passing
            edge_index_dict = batch.edge_index_dict
            
            for i, conv in enumerate(self.convs):
                # Apply convolution for each edge type
                x_dict_new = {}
                for node_type in self.node_types:
                    # Aggregate messages from all edge types involving this node
                    msgs = []
                    for edge_type in self.edge_types:
                        src, rel, dst = edge_type
                        if dst == node_type and edge_type in edge_index_dict:
                            edge_index = edge_index_dict[edge_type]
                            msg = conv((x_dict[src], x_dict[dst]), edge_index)
                            msgs.append(msg)
                    
                    if msgs:
                        x_dict_new[node_type] = torch.stack(msgs).mean(dim=0).relu()
                    else:
                        x_dict_new[node_type] = x_dict[node_type]
                    
                    # Add to output with weight
                    out_dict[node_type] = out_dict[node_type] + x_dict_new[node_type] * weights[i + 1]
                
                x_dict = x_dict_new
        
        elif self.model_type == "HGT":
            # Heterogeneous Graph Transformer
            edge_index_dict = batch.edge_index_dict
            
            for i, conv in enumerate(self.convs):
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
                
                # Add to output with weight
                for node_type in self.node_types:
                    out_dict[node_type] = out_dict[node_type] + x_dict[node_type] * weights[i + 1]
        
        return out_dict

    def _to_homogeneous(self, batch):
        """Convert heterogeneous batch to homogeneous for RGCN."""
        edge_indices = []
        edge_types = []
        
        # Create node offset mapping
        node_offset = {}
        offset = 0
        for node_type in self.node_types:
            node_offset[node_type] = offset
            offset += batch[node_type].num_nodes
        
        # Collect all edges with their types
        for edge_type_idx, edge_type in enumerate(self.edge_types):
            src_type, _, dst_type = edge_type
            if edge_type in batch.edge_index_dict:
                edge_index = batch.edge_index_dict[edge_type]
                # Offset node indices
                edge_index_offset = edge_index.clone()
                edge_index_offset[0] += node_offset[src_type]
                edge_index_offset[1] += node_offset[dst_type]
                
                edge_indices.append(edge_index_offset)
                edge_types.append(torch.full((edge_index.size(1),), edge_type_idx, dtype=torch.long))
        
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_types, dim=0)
        
        return edge_index.to(batch['playlist'].n_id.device), edge_type.to(batch['playlist'].n_id.device)

    def forward(self, batch, edge_label_index: OptTensor = None) -> Tensor:
        """
        Forward pass for link prediction.
        
        Args:
            batch: HeteroData batch
            edge_label_index: [2, num_edges] edges to predict (playlist, track)
            
        Returns:
            Prediction scores for edges
        """
        if edge_label_index is None:
            edge_label_index = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label_index
        
        out_dict = self.get_embedding(batch)
        
        return self.predict_link_embedding(out_dict, edge_label_index, batch)

    def predict_link_embedding(self, embed_dict: Dict[str, Tensor], edge_label_index: Tensor, 
                              batch=None) -> Tensor:
        """
        Predict link scores using dot product.
        
        Args:
            embed_dict: Dictionary of embeddings per node type (LOCAL batch embeddings)
            edge_label_index: [2, num_edges] (playlist_global_idx, track_global_idx)
            batch: HeteroData batch for global-to-local mapping
            
        Returns:
            scores: [num_edges] prediction scores
        """
        # embed_dict contains embeddings indexed by LOCAL batch indices
        # edge_label_index contains GLOBAL node IDs
        # We need to map global IDs to local batch indices
        
        if batch is None:
            # No batch provided, assume edge_label_index has local indices
            playlist_emb = embed_dict['playlist']
            track_emb = embed_dict['track']
            embed_src = playlist_emb[edge_label_index[0]]
            embed_dst = track_emb[edge_label_index[1]]
        else:
            # Get full embedding tables and index directly by global ID
            playlist_emb_full = self.embeddings['playlist'].weight
            track_emb_full = self.embeddings['track'].weight
            
            # But we need the message-passed embeddings from embed_dict
            # Solution: index the full embeddings, then apply the final layer output
            # Actually, let's just use batch n_id to map properly
            
            playlist_n_id = batch['playlist'].n_id
            track_n_id = batch['track'].n_id
            
            # Create mapping
            # Find local index for each global ID in edge_label_index
            playlist_global_ids = edge_label_index[0]
            track_global_ids = edge_label_index[1]
            
            # Use searchsorted for efficient mapping (requires sorted n_id)
            playlist_sorted, playlist_sort_idx = playlist_n_id.sort()
            track_sorted, track_sort_idx = track_n_id.sort()
            
            playlist_pos = torch.searchsorted(playlist_sorted, playlist_global_ids.to(device="cuda"))
            track_pos = torch.searchsorted(track_sorted, track_global_ids.to(device="cuda"))
            
            # Get actual local indices
            playlist_local = playlist_sort_idx[playlist_pos]
            track_local = track_sort_idx[track_pos]
            
            # Index embeddings
            embed_src = embed_dict['playlist'][playlist_local]
            embed_dst = embed_dict['track'][track_local]
        
        return (embed_src * embed_dst).sum(dim=-1)

    def predict_link(self, batch, edge_label_index: OptTensor = None, prob: bool = False) -> Tensor:
        """Predict links with optional probability output."""
        pred = self(batch, edge_label_index).sigmoid()
        return pred if prob else pred.round()

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        """Binary cross-entropy loss for link prediction."""
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def bpr_loss(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """
        Bayesian Personalized Ranking loss.
        Handles multiple negatives per positive.
        """
        num_pos = pos_scores.size(0)
        num_neg = neg_scores.size(0)
        
        if num_pos == num_neg:
            # Equal number of positives and negatives
            return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        else:
            # Multiple negatives per positive - expand positives
            neg_ratio = num_neg // num_pos
            pos_scores_expanded = pos_scores.repeat_interleave(neg_ratio)
            return -torch.log(torch.sigmoid(pos_scores_expanded - neg_scores)).mean()

    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                           lambda_reg: float = 1e-4) -> Tensor:
        """BPR loss with L2 regularization."""
        bpr = self.bpr_loss(pos_edge_rank, neg_edge_rank)
        
        # L2 regularization on embeddings
        reg_loss = 0
        for emb in self.embeddings.values():
            reg_loss += emb.weight.norm(2).pow(2)
        
        return bpr + lambda_reg * reg_loss

    def __repr__(self) -> str:
        total_nodes = sum(self.num_nodes_dict.values())
        return (f'{self.__class__.__name__}({self.model_type}, '
                f'nodes={total_nodes}, '
                f'emb_dim={self.embedding_dim}, '
                f'layers={self.num_layers})')


# ============================================================================
# Usage Examples
# ============================================================================

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


# Example usage:
# metadata = train_split.metadata()
# num_nodes_dict = {
#     'playlist': 10000,
#     'track': 171855,
#     'album': 81720,
#     'artist': 35797
# }

# # Create model
# device = "cpu"
# model = create_model_rgcn(metadata, num_nodes_dict, embedding_dim=128, num_layers=3)
# model = model.to(device)

# # Training loop
# for batch in train_loader:
#     batch = batch.to(device)
    
#     # Forward pass
#     pos_pred = model(batch)
    
#     # Sample negatives
#     neg_edge_index, neg_edge_label = sample_negatives(batch)
#     neg_pred = model(batch, neg_edge_index)
    
#     # Compute loss
#     loss = model.bpr_loss(pos_pred, neg_pred)
    
#     # Backward
#     loss.backward()
#     optimizer.step()

def train_model(model, train_loader, optimizer, device):
    model.train()
    scaler = amp.GradScaler()  # Initialize the gradient scaler for mixed precision

    for batch in train_loader:
        batch = batch.to(device)

        with amp.autocast():  # Enable mixed precision
            pos_pred = model(batch)
            neg_edge_index, neg_edge_label = sample_negatives(batch)
            neg_pred = model(batch, neg_edge_index)
            loss = model.bpr_loss(pos_pred, neg_pred)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # Scale the loss and call backward
        scaler.step(optimizer)  # Update the parameters
        scaler.update()  # Update the scale for next iteration