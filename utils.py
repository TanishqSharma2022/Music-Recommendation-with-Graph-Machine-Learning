import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import Tensor


# def sample_negatives(batch, neg_ratio=3):
#     """
#     Sample random negative edges for playlist-track link prediction.
    
#     Args:
#         batch: HeteroData batch from neighbor loader
#         neg_ratio: Number of negatives per positive edge (default: 3)
        
#     Returns:
#         neg_edge_index: [2, num_neg] negative edges (playlist_idx, track_idx)
#         neg_edge_label: [num_neg] all zeros
#     """
#     # Get positive edges and batch nodes
#     pos_edge_index = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label_index
#     batch_playlists = batch['playlist'].n_id
#     batch_tracks = batch['track'].n_id
    
#     num_pos = pos_edge_index.size(1)
#     num_neg = num_pos * neg_ratio
    
#     # Create set of positive edges for fast lookup
#     pos_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    
#     # Sample negatives
#     neg_edges = []
#     while len(neg_edges) < num_neg:
#         # Random playlist and track from batch
#         p_idx = batch_playlists[torch.randint(0, len(batch_playlists), (1,))].item()
#         t_idx = batch_tracks[torch.randint(0, len(batch_tracks), (1,))].item()
        
#         # Keep if not a positive edge
#         if (p_idx, t_idx) not in pos_set:
#             neg_edges.append([p_idx, t_idx])
    
#     neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().to(pos_edge_index.device)
#     neg_edge_label = torch.zeros(num_neg, dtype=torch.float, device=pos_edge_index.device)
    
#     return neg_edge_index, neg_edge_label

def sample_negatives(batch, device="cpu", neg_ratio=3):
    """Sample negative edges within batch node space."""
    edge_type = ('playlist', 'track_in_playlist', 'track')
    
    # Get positive edges and their label info
    pos_edge_index = batch[edge_type].pos_edge_label_index
    pos_edge_label = batch[edge_type].pos_edge_label
    
    # Get valid node ranges in THIS batch
    num_playlists = batch['playlist'].num_nodes
    num_tracks = batch['track'].num_nodes
    
    # Create set of positive edges to avoid duplicates
    pos_set = set(zip(pos_edge_index[0].cpu().tolist(), pos_edge_index[1].cpu().tolist()))
    
    neg_edges = []
    num_negatives = len(pos_edge_label) * neg_ratio
    
    # Sample negative edges from valid node range
    while len(neg_edges) < num_negatives:
        # Sample random playlist and track IDs from BATCH range
        playlist_id = torch.randint(0, num_playlists, (1,)).item()
        track_id = torch.randint(0, num_tracks, (1,)).item()
        
        # Skip if already a positive edge
        if (playlist_id, track_id) not in pos_set:
            neg_edges.append([playlist_id, track_id])
    
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(device)
    neg_edge_label = torch.zeros(len(neg_edges), dtype=torch.long).to(device)
    
    return neg_edge_index, neg_edge_label


def sample_hard_negatives(batch, model, device=None, batch_size=500, frac_sample=1.0):
    """
    Sample hard negative edges based on model embeddings.
    
    Args:
        batch: HeteroData batch from neighbor loader
        model: Your GNN model with get_embedding() method
        device: Device to use (default: same as batch)
        batch_size: Batch size for scoring (default: 500)
        frac_sample: Fraction of tracks to consider for sampling (default: 1.0)
        
    Returns:
        neg_edge_index: [2, num_neg] hard negative edges
        neg_edge_label: [num_neg] all zeros
    """
    if device is None:
        device = batch['playlist'].n_id.device
    
    with torch.no_grad():
        # Get embeddings from model using the batch
        embed_dict = model.get_embedding(batch)
        
        # Extract playlist and track embeddings (LOCAL batch embeddings)
        playlist_emb = embed_dict['playlist'].to(device)
        track_emb = embed_dict['track'].to(device)
        
        # Get positive edges (GLOBAL IDs)
        pos_edge_index = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label_index
        positive_playlists_global = pos_edge_index[0]
        positive_tracks_global = pos_edge_index[1]
        num_edges = positive_playlists_global.size(0)
        
        # Get batch node mappings
        batch_playlists = batch['playlist'].n_id
        batch_tracks = batch['track'].n_id
        num_batch_playlists = len(batch_playlists)
        num_batch_tracks = len(batch_tracks)
        
        # Create reverse mappings: global_id → local_idx
        playlist_map = {global_id.item(): local_idx for local_idx, global_id in enumerate(batch_playlists.cpu())}
        track_map = {global_id.item(): local_idx for local_idx, global_id in enumerate(batch_tracks.cpu())}
        
        # Map positive edges from global to local indices
        pos_playlists_local = torch.tensor(
            [playlist_map[pid.item()] for pid in positive_playlists_global.cpu()],
            dtype=torch.long,
            device=device
        )
        pos_tracks_local = torch.tensor(
            [track_map[tid.item()] for tid in positive_tracks_global.cpu()],
            dtype=torch.long,
            device=device
        )
        
        # Create positive edge mask (LOCAL indices)
        positive_mask = torch.zeros(num_batch_playlists, num_batch_tracks, device=device, dtype=torch.bool)
        positive_mask[pos_playlists_local, pos_tracks_local] = True
        
        neg_edges_list = []
        neg_edge_label_list = []
        
        # Process in batches
        for batch_start in range(0, num_edges, batch_size):
            batch_end = min(batch_start + batch_size, num_edges)
            
            # Get local playlist indices for this batch
            playlists_local_batch = pos_playlists_local[batch_start:batch_end]
            
            # Compute similarity scores (using LOCAL embeddings)
            batch_scores = torch.matmul(
                playlist_emb[playlists_local_batch], 
                track_emb.t()
            )
            
            # Mask out positive edges
            batch_scores[positive_mask[playlists_local_batch]] = -float("inf")
            
            # Select top-k highest scoring negative edges
            k = int(frac_sample * 0.99 * num_batch_tracks)
            k = max(1, k)  # Ensure at least 1
            _, top_indices_local = torch.topk(batch_scores, k, dim=1)
            
            # Randomly select one from top-k for each playlist
            selected_indices = torch.randint(0, k, size=(batch_end - batch_start,), device=device)
            top_tracks_local = top_indices_local[torch.arange(batch_end - batch_start), selected_indices]
            
            # Map local indices back to global for output
            playlists_global_batch = positive_playlists_global[batch_start:batch_end]
            tracks_global_batch = batch_tracks[top_tracks_local]
            
            # Create negative edges (GLOBAL IDs)
            neg_edges_batch = torch.stack(
                (playlists_global_batch, tracks_global_batch), dim=0
            )
            neg_edge_label_batch = torch.zeros(neg_edges_batch.shape[1], device=device)
            
            neg_edges_list.append(neg_edges_batch)
            neg_edge_label_list.append(neg_edge_label_batch)
        
        # Concatenate all batches
        neg_edge_index = torch.cat(neg_edges_list, dim=1)
        neg_edge_label = torch.cat(neg_edge_label_list)
        
        return neg_edge_index, neg_edge_label


def recall_at_k(batch, model, k=300, score_batch_size=64, device=None):
    """
    Calculate Recall@K for heterogeneous batch.
    
    Args:
        batch: HeteroData batch from neighbor loader
        model: Your GNN model with get_embedding() method
        k: Top-k items to consider (default: 300)
        score_batch_size: Batch size for scoring (default: 64)
        device: Device to use
        
    Returns:
        recall_at_k: Scalar recall value
    """
    if device is None:
        device = batch['playlist'].n_id.device
    
    with torch.no_grad():
        # Get embeddings from model using the batch
        embed_dict = model.get_embedding(batch)
        
        # Extract LOCAL batch embeddings
        playlist_emb = embed_dict['playlist'].to(device)
        track_emb = embed_dict['track'].to(device)
        
        num_batch_playlists = playlist_emb.size(0)
        num_batch_tracks = track_emb.size(0)
        
        # Get edges (GLOBAL IDs)
        mp_edge_index = batch['playlist', 'track_in_playlist', 'track'].edge_index
        gt_edge_index = batch['playlist', 'track_in_playlist', 'track'].pos_edge_label_index
        
        # Get batch node mappings
        batch_tracks = batch['track'].n_id
        batch_playlists = batch['playlist'].n_id
        
        # Create reverse mappings: global_id → local_idx
        track_global_to_local = {t.item(): i for i, t in enumerate(batch_tracks.cpu())}
        playlist_global_to_local = {p.item(): i for i, p in enumerate(batch_playlists.cpu())}
        
        hits_list = []
        relevant_counts_list = []
        
        # Process playlists in batches
        for batch_start in range(0, num_batch_playlists, score_batch_size):
            batch_end = min(batch_start + score_batch_size, num_batch_playlists)
            batch_playlist_emb = playlist_emb[batch_start:batch_end]
            
            # Calculate scores for all tracks (using LOCAL embeddings)
            scores = torch.matmul(batch_playlist_emb, track_emb.t())
            
            # Mask out message passing edges (exclude training edges)
            for i in range(mp_edge_index.size(1)):
                p_global = mp_edge_index[0, i].item()
                t_global = mp_edge_index[1, i].item()
                
                # Check if both nodes are in this batch
                if p_global in playlist_global_to_local and t_global in track_global_to_local:
                    p_local = playlist_global_to_local[p_global]
                    t_local = track_global_to_local[t_global]
                    
                    # Check if playlist is in current scoring batch
                    if batch_start <= p_local < batch_end:
                        scores[p_local - batch_start, t_local] = -float("inf")
            
            # Get top-k predictions
            actual_k = min(k, num_batch_tracks)
            _, top_k_indices = torch.topk(scores, actual_k, dim=1)
            
            # Create ground truth mask (LOCAL indices)
            mask = torch.zeros(scores.shape, device=device, dtype=torch.bool)
            
            for i in range(gt_edge_index.size(1)):
                p_global = gt_edge_index[0, i].item()
                t_global = gt_edge_index[1, i].item()
                
                # Check if both nodes are in this batch
                if p_global in playlist_global_to_local and t_global in track_global_to_local:
                    p_local = playlist_global_to_local[p_global]
                    t_local = track_global_to_local[t_global]
                    
                    # Check if playlist is in current scoring batch
                    if batch_start <= p_local < batch_end:
                        mask[p_local - batch_start, t_local] = True
            
            # Count hits (how many ground truth items are in top-k)
            hits = mask.gather(1, top_k_indices).sum(dim=1)
            hits_list.append(hits)
            
            # Count total relevant items per playlist in this scoring batch
            relevant_counts = torch.zeros(batch_end - batch_start, device=device)
            for i in range(gt_edge_index.size(1)):
                p_global = gt_edge_index[0, i].item()
                
                if p_global in playlist_global_to_local:
                    p_local = playlist_global_to_local[p_global]
                    if batch_start <= p_local < batch_end:
                        relevant_counts[p_local - batch_start] += 1
            
            relevant_counts_list.append(relevant_counts)
        
        # Compute recall@k
        hits_tensor = torch.cat(hits_list, dim=0)
        relevant_counts_tensor = torch.cat(relevant_counts_list, dim=0)
        
        # Handle division by zero (playlists with no ground truth edges)
        recall_at_k = torch.where(
            relevant_counts_tensor != 0,
            hits_tensor.float() / relevant_counts_tensor,
            torch.zeros_like(hits_tensor, dtype=torch.float)  # Changed from ones to zeros
        )
        
        # Average recall across all playlists
        recall_at_k = torch.mean(recall_at_k)
        
        return recall_at_k.item()

class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).sum()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs
    


def recall_at_k_fast(batch, model, k=300, score_batch_size=512, device=None):
    import torch

    if device is None:
        device = batch["playlist"].n_id.device

    with torch.no_grad():
        embed = model.get_embedding(batch)
        pl_emb = embed["playlist"]
        tr_emb = embed["track"]

        num_pl = pl_emb.size(0)
        num_tr = tr_emb.size(0)

        # global → local index maps
        pl_local = batch["playlist"].n_id
        tr_local = batch["track"].n_id

        # --- Build LOCAL edge indices (fast vectorized way) ---
        # map globals to locals
        mp = batch["playlist", "track_in_playlist", "track"].edge_index
        gt = batch["playlist", "track_in_playlist", "track"].pos_edge_label_index

        mp_pl = torch.searchsorted(pl_local, mp[0])
        mp_tr = torch.searchsorted(tr_local, mp[1])

        gt_pl = torch.searchsorted(pl_local, gt[0])
        gt_tr = torch.searchsorted(tr_local, gt[1])

        # mask for valid edges (in this neighbor batch)
        valid_mp = (
            (mp_pl >= 0) & (mp_pl < num_pl) &
            (mp_tr >= 0) & (mp_tr < num_tr)
        )
        valid_gt = (
            (gt_pl >= 0) & (gt_pl < num_pl) &
            (gt_tr >= 0) & (gt_tr < num_tr)
        )

        mp_pl = mp_pl[valid_mp]
        mp_tr = mp_tr[valid_mp]
        gt_pl = gt_pl[valid_gt]
        gt_tr = gt_tr[valid_gt]

        # Build full boolean ground-truth matrix (playlist × track)
        # Much faster than per-playlist mask
        gt_matrix = torch.zeros((num_pl, num_tr), dtype=torch.bool, device=device)
        gt_matrix[gt_pl, gt_tr] = True

        # Build mask of edges to exclude (train edges)
        exclude_matrix = torch.zeros((num_pl, num_tr), dtype=torch.bool, device=device)
        exclude_matrix[mp_pl, mp_tr] = True

        hits_total = []
        relevant_total = []

        # batch scoring
        for start in range(0, num_pl, score_batch_size):
            end = min(start + score_batch_size, num_pl)

            scores = pl_emb[start:end] @ tr_emb.t()

            # Remove training edges
            scores[exclude_matrix[start:end]] = -float("inf")

            # top-k
            k_eff = min(k, num_tr)
            _, topk = torch.topk(scores, k_eff, dim=1)

            # compute hits by gathering
            hits = gt_matrix[start:end].gather(1, topk).sum(dim=1)
            hits_total.append(hits)

            relevant_total.append(gt_matrix[start:end].sum(dim=1))

        hits_total = torch.cat(hits_total)
        relevant_total = torch.cat(relevant_total)

        recall = torch.where(
            relevant_total > 0,
            hits_total.float() / relevant_total,
            torch.zeros_like(hits_total, dtype=torch.float)
        )

        return recall.mean().item()
