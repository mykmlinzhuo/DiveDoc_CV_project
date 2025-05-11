import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

class HierarchicalSkeletalEncoder(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,          # (x, y) coordinates
        hidden_dims: tuple = (64, 64, 64),
        edge_hidden: int = 64,
        out_dim: int = 128
    ):
        super().__init__()
        self.num_joints = num_joints

        # Define semantic-level subsets (COCO indices):
        # H0: core (hips, shoulders)
        # H1: connectors (elbows, knees)
        # H2: extremities (wrists, ankles)
        self.subsets = [
            [0, 5, 6, 11, 12],       # H0: Core
            [7, 8, 13, 14],          # H1: Connectors
            [9, 10, 15, 16]          # H2: Extremities
        ]

        # Build physical adjacency per level (complete within-subset graph)
        adjs = []
        for subset in self.subsets:
            A = torch.zeros(num_joints, num_joints)
            for u in subset:
                for v in subset:
                    if u != v:
                        A[u, v] = 1
            adjs.append(A)
        # Register as buffers so they move with .to(device)
        for i, A in enumerate(adjs):
            self.register_buffer(f'adj_phys_{i}', A)

        # Build cross-level adjacency (between adjacent semantic levels)
        cross_adjs = []
        for i, subset in enumerate(self.subsets):
            C = torch.zeros(num_joints, num_joints)
            if i > 0:
                for u in subset:
                    for v in self.subsets[i - 1]:
                        C[u, v] = C[v, u] = 1
            if i < len(self.subsets) - 1:
                for u in subset:
                    for v in self.subsets[i + 1]:
                        C[u, v] = C[v, u] = 1
            cross_adjs.append(C)
        for i, C in enumerate(cross_adjs):
            self.register_buffer(f'adj_cross_{i}', C)

        # Per-level MLP for node features (coords + confidence)
        self.level_mlps = nn.ModuleList([
            nn.Linear(input_dim + 1, hidden_dims[i]) for i in range(3)
        ])

        # EdgeConv MLP: input is [h_j*c_j ∥ (h_k*c_k - h_j*c_j)]
        self.edge_mlp = nn.Linear(2 * hidden_dims[1], edge_hidden)

        # Final projection to out_dim
        # We concatenate for each level: mean(H) + mean(Z) → total = sum(hidden_dims) + 3*edge_hidden
        total_dim = sum(hidden_dims) + 3 * edge_hidden
        self.proj = nn.Linear(total_dim, out_dim)

    def forward(self, detections: List[dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            detections: list of length N, each either None or a dict:
                {
                  'keypoints': Tensor[17,2],   # (x,y)
                  'scores':    Tensor[17]      # confidences
                }
        Returns:
            Tensor of shape [N, out_dim]
        """
        device = None
        for det in detections:
            if det is not None:
                device = det["keypoints"].device
                break
        if device is None:
            device = next(self.proj.parameters()).device
        
        out_feats = []
        for det in detections:
            if det is None:
                # If no detection, output zero feature
                out_feats.append(torch.zeros(self.proj.out_features, device=device))
                continue

            kpts = det['keypoints'].to(device)       # Tensor[17,2]
            scores = det['scores'].unsqueeze(-1).to(device)   # [17,1]

            # STEP 1: Normalize keypoints
            # Normalize x, y coordinates to [0, 1] based on the bounding box of the keypoints
            min_vals, _ = kpts.min(dim=0, keepdim=True)  # [1, 2]
            max_vals, _ = kpts.max(dim=0, keepdim=True)  # [1, 2]
            kpts = (kpts - min_vals) / (max_vals - min_vals + 1e-6)  # Avoid division by zero


            # Build input: [x, y, confidence]
            P = torch.cat([kpts, scores], dim=-1)  # [17, 3]

            # STEP 2: Confidence-aware GCN per level
            H_levels = []
            for i in range(3):
                A_phys = getattr(self, f'adj_phys_{i}')
                h = F.relu(self.level_mlps[i](P))      # [17, hidden_i]
                m = h * scores                         # weight by confidence
                H = torch.matmul(A_phys, m)           # [17, hidden_i]
                H_levels.append(H)

            # STEP 3: EdgeConv on each level
            Z_levels = []
            for i, H in enumerate(H_levels):
                A_phys = getattr(self, f'adj_phys_{i}')
                A_cross = getattr(self, f'adj_cross_{i}')
                A_exp = A_phys + A_cross              # expanded adjacency
                edge_msgs = []
                for j in range(self.num_joints):
                    nbrs = (A_exp[j] > 0).nonzero(as_tuple=False).squeeze(-1)
                    msgs = []
                    for k in nbrs.tolist():
                        feat_j = H[j] * scores[j]
                        feat_k = H[k] * scores[k]
                        edge_feat = torch.cat([feat_j, (feat_k - feat_j)], dim=-1)
                        msgs.append(F.relu(self.edge_mlp(edge_feat)))
                    if msgs:
                        msgs = torch.stack(msgs, dim=0)
                        edge_msgs.append(msgs.max(dim=0)[0])
                    else:
                        edge_msgs.append(torch.zeros(self.edge_mlp.out_features, device=device))
                Z = torch.stack(edge_msgs, dim=0)     # [17, edge_hidden]
                Z_levels.append(Z)

            # POOL & PROJECTION
            pooled = []
            for H, Z in zip(H_levels, Z_levels):
                F_i = torch.cat([H, Z], dim=-1)       # [17, hidden_i + edge_hidden]
                pooled.append(F_i.mean(dim=0))        # [hidden_i + edge_hidden]
            final_feat = torch.cat(pooled, dim=-1)    # [sum(hidden_i) + 3*edge_hidden]
            out_feats.append(self.proj(final_feat))   # [out_dim]

        return torch.stack(out_feats, dim=0).to(device)         # [N, out_dim]



class OptimizedHierarchicalEncoder(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,          # (x, y) coordinates
        hidden_dims: tuple = (64, 64, 64),
        edge_hidden: int = 64,
        out_dim: int = 128
    ):
        super().__init__()
        self.num_joints = num_joints

        # Define semantic-level subsets (COCO indices)
        self.subsets = [
            [0, 5, 6, 11, 12],       # H0: Core
            [7, 8, 13, 14],          # H1: Connectors
            [9, 10, 15, 16]          # H2: Extremities
        ]

        # Pre-compute masks for more efficient operations
        self._create_adjacency_masks(num_joints)

        # Per-level MLP for node features (coords + confidence)
        self.level_mlps = nn.ModuleList([
            nn.Linear(input_dim + 1, hidden_dims[i]) for i in range(3)
        ])

        # EdgeConv MLP: input is [h_j*c_j ∥ (h_k*c_k - h_j*c_j)]
        self.edge_mlp = nn.Linear(2 * hidden_dims[1], edge_hidden)

        # Final projection to out_dim
        total_dim = sum(hidden_dims) + 3 * edge_hidden
        self.proj = nn.Linear(total_dim, out_dim)

    def _create_adjacency_masks(self, num_joints):
        """Pre-compute adjacency masks for efficient operations."""
        # Build masks for each level
        for i, subset in enumerate(self.subsets):
            # Physical adjacency (within-level)
            mask = torch.zeros(num_joints, dtype=torch.bool)
            mask[subset] = True
            self.register_buffer(f'mask_{i}', mask)
            
            # Cross-level connections
            if i > 0:  # Connect to previous level
                prev_mask = torch.zeros(num_joints, dtype=torch.bool)
                prev_mask[self.subsets[i-1]] = True
                self.register_buffer(f'prev_mask_{i}', prev_mask)
            if i < len(self.subsets) - 1:  # Connect to next level
                next_mask = torch.zeros(num_joints, dtype=torch.bool)
                next_mask[self.subsets[i+1]] = True
                self.register_buffer(f'next_mask_{i}', next_mask)

    def _batch_edge_conv(self, H, scores, level_idx):
        """Vectorized EdgeConv operation."""
        device = H.device
        num_joints = self.num_joints
        
        # Get masks for current level
        curr_mask = getattr(self, f'mask_{level_idx}')
        
        # Initialize edge features tensor
        edge_features = torch.zeros(num_joints, self.edge_mlp.out_features, device=device)
        
        # Process each level
        curr_joints = curr_mask.nonzero(as_tuple=True)[0]
        
        # Get neighboring joints (both within level and cross-level)
        neighbor_masks = [curr_mask]
        if hasattr(self, f'prev_mask_{level_idx}'):
            neighbor_masks.append(getattr(self, f'prev_mask_{level_idx}'))
        if hasattr(self, f'next_mask_{level_idx}'):
            neighbor_masks.append(getattr(self, f'next_mask_{level_idx}'))
        
        # Combine all neighbor masks
        all_neighbors = torch.zeros(num_joints, dtype=torch.bool, device=device)
        for mask in neighbor_masks:
            all_neighbors = all_neighbors | mask
        
        # For each joint in current level
        for j in curr_joints:
            # Get weighted feature for source joint
            feat_j = H[j] * scores[j, 0]  # Extract scalar from [1] tensor
            
            # Find neighbors
            neighbors = all_neighbors.nonzero(as_tuple=True)[0]
            neighbors = neighbors[neighbors != j]  # Remove self-loop
            
            if len(neighbors) == 0:
                continue
                
            # Get weighted features for all neighbors
            feat_k = H[neighbors] * scores[neighbors, 0].unsqueeze(-1)  # [n_neighbors, hidden_dim]
            
            # Expand feat_j to match shape of feat_k
            feat_j_expanded = feat_j.unsqueeze(0).expand(len(neighbors), -1)  # [n_neighbors, hidden_dim]
            
            # Concatenate source and difference features
            edge_feat = torch.cat([feat_j_expanded, (feat_k - feat_j_expanded)], dim=-1)
            
            # Apply edge MLP
            edge_msgs = F.relu(self.edge_mlp(edge_feat))
            
            # Max pooling over neighbors
            if len(edge_msgs) > 0:
                edge_features[j] = torch.max(edge_msgs, dim=0)[0]
        
        return edge_features

    def forward(self, detections: List[Union[dict, None]]) -> torch.Tensor:
        """
        Args:
            detections: list of length N, each either None or a dict:
                {
                  'keypoints': Tensor[17,2],   # (x,y)
                  'scores':    Tensor[17]      # confidences
                }
        Returns:
            Tensor of shape [N, out_dim]
        """
        # Get device once
        device = next(self.parameters()).device
        
        out_features = []
        for det in detections:
            if det is None:
                # If no detection, output zero feature
                out_features.append(torch.zeros(self.proj.out_features, device=device))
                continue

            # Move tensors to device only once
            kpts = det['keypoints'].to(device)  # [17, 2]
            scores = det['scores'].unsqueeze(-1).to(device)  # [17, 1]

            # STEP 1: Normalize keypoints
            min_vals, _ = kpts.min(dim=0, keepdim=True)
            max_vals, _ = kpts.max(dim=0, keepdim=True)
            kpts = (kpts - min_vals) / (max_vals - min_vals + 1e-6)

            # Build input: [x, y, confidence]
            P = torch.cat([kpts, scores], dim=-1)  # [17, 3]

            # STEP 2: Process each level efficiently
            H_levels = []
            Z_levels = []
            
            for i in range(3):
                # Get mask for current level
                curr_mask = getattr(self, f'mask_{i}')
                
                # Apply MLP to all joints
                h = F.relu(self.level_mlps[i](P))  # [17, hidden_i]
                m = h * scores  # Weight by confidence
                
                # Use mask to select relevant joints for this level
                curr_H = torch.zeros_like(h)
                
                # For each joint in the level, aggregate features from all other joints in the level
                level_joints = curr_mask.nonzero(as_tuple=True)[0]
                for j in level_joints:
                    # Aggregate features from all other joints in this level
                    neighbors = curr_mask.nonzero(as_tuple=True)[0]
                    if len(neighbors) > 0:
                        curr_H[j] = m[neighbors].sum(dim=0)
                
                H_levels.append(curr_H)
                
                # Perform batch edge convolution
                Z = self._batch_edge_conv(h, scores, i)
                Z_levels.append(Z)
            
            # POOL & PROJECTION
            pooled = []
            for i, (H, Z) in enumerate(zip(H_levels, Z_levels)):
                # Get mask for current level
                mask = getattr(self, f'mask_{i}')
                level_joints = mask.nonzero(as_tuple=True)[0]
                
                # Only pool features from joints in this level
                if len(level_joints) > 0:
                    # Concatenate node and edge features
                    F_i = torch.cat([H[level_joints], Z[level_joints]], dim=-1)
                    # Mean pooling
                    pooled.append(F_i.mean(dim=0))
                else:
                    # If no joints in this level, use zeros
                    pooled.append(torch.zeros(H.shape[1] + Z.shape[1], device=device))
            
            # Concatenate all levels
            final_feat = torch.cat(pooled, dim=-1)
            
            # Project to output dimension
            out_features.append(self.proj(final_feat))

        return torch.stack(out_features, dim=0)

class FullyOptimizedHierarchicalEncoder(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,          # (x, y) coordinates
        hidden_dims: tuple = (64, 64, 64),
        edge_hidden: int = 64,
        out_dim: int = 128
    ):
        super().__init__()
        self.num_joints = num_joints

        # Define semantic-level subsets (COCO indices)
        self.subsets = [
            [0, 5, 6, 11, 12],       # H0: Core
            [7, 8, 13, 14],          # H1: Connectors
            [9, 10, 15, 16]          # H2: Extremities
        ]

        # Pre-compute adjacency matrices and store as buffers
        self._precompute_adjacencies(num_joints)

        # Per-level MLP for node features (coords + confidence)
        self.level_mlps = nn.ModuleList([
            nn.Linear(input_dim + 1, hidden_dims[i]) for i in range(3)
        ])

        # EdgeConv MLP: input is [h_j*c_j ∥ (h_k*c_k - h_j*c_j)]
        self.edge_mlp = nn.Linear(2 * hidden_dims[0], edge_hidden)  # Use same hidden dim for all levels

        # Final projection to out_dim
        total_dim = sum(hidden_dims) + 3 * edge_hidden
        self.proj = nn.Linear(total_dim, out_dim)

    def _precompute_adjacencies(self, num_joints):
        """Pre-compute all adjacency matrices for faster processing."""
        # Create joint masks for each level
        level_masks = []
        for subset in self.subsets:
            mask = torch.zeros(num_joints, dtype=torch.bool)
            mask[subset] = True
            level_masks.append(mask)
        
        # Register level masks
        for i, mask in enumerate(level_masks):
            self.register_buffer(f'level_mask_{i}', mask)
        
        # For each level, create:
        # 1. Physical adjacency matrix (within level)
        # 2. Cross-level adjacency matrix
        # 3. Combined adjacency matrix
        for i, subset in enumerate(self.subsets):
            # Physical adjacency (fully connected within level)
            phys_adj = torch.zeros(num_joints, num_joints)
            for u in subset:
                for v in subset:
                    if u != v:
                        phys_adj[u, v] = 1
            
            # Cross-level adjacency
            cross_adj = torch.zeros(num_joints, num_joints)
            if i > 0:  # Connect to previous level
                for u in subset:
                    for v in self.subsets[i-1]:
                        cross_adj[u, v] = cross_adj[v, u] = 1
            if i < len(self.subsets) - 1:  # Connect to next level
                for u in subset:
                    for v in self.subsets[i+1]:
                        cross_adj[u, v] = cross_adj[v, u] = 1
            
            # Combined adjacency
            combined_adj = phys_adj + cross_adj
            
            # Register as buffers
            self.register_buffer(f'phys_adj_{i}', phys_adj)
            self.register_buffer(f'cross_adj_{i}', cross_adj)
            self.register_buffer(f'combined_adj_{i}', combined_adj)

    def forward(self, detections: List[Union[dict, None]]) -> torch.Tensor:
        """
        Fully vectorized forward pass with minimal loops.
        """
        device = next(self.parameters()).device
        batch_size = len(detections)
        
        # Prepare output tensor
        output = torch.zeros(batch_size, self.proj.out_features, device=device)
        
        # Process each detection in the batch
        valid_indices = []
        valid_inputs = []
        valid_scores = []
        
        for i, det in enumerate(detections):
            if det is not None:
                valid_indices.append(i)
                valid_inputs.append(det['keypoints'].to(device))
                valid_scores.append(det['scores'].unsqueeze(-1).to(device))
        
        # If no valid detections, return zeros
        if not valid_indices:
            return output
        
        # Stack all valid inputs
        kpts = torch.stack(valid_inputs)  # [valid_batch, 17, 2]
        scores = torch.stack(valid_scores)  # [valid_batch, 17, 1]
        
        # Normalize keypoints (vectorized for all valid samples)
        min_vals = kpts.min(dim=1, keepdim=True)[0]  # [valid_batch, 1, 2]
        max_vals = kpts.max(dim=1, keepdim=True)[0]  # [valid_batch, 1, 2]
        kpts_norm = (kpts - min_vals) / (max_vals - min_vals + 1e-6)  # [valid_batch, 17, 2]
        
        # Create input features [x, y, confidence]
        P = torch.cat([kpts_norm, scores], dim=-1)  # [valid_batch, 17, 3]
        
        # Process each level
        all_H_pooled = []
        all_Z_pooled = []
        
        for level_idx in range(3):
            # Get adjacency matrices for this level
            phys_adj = getattr(self, f'phys_adj_{level_idx}')
            combined_adj = getattr(self, f'combined_adj_{level_idx}')
            level_mask = getattr(self, f'level_mask_{level_idx}')
            
            # Apply MLP to all inputs
            H = F.relu(self.level_mlps[level_idx](P))  # [valid_batch, 17, hidden_dim]
            
            # Weight by confidence scores
            H_weighted = H * scores  # [valid_batch, 17, hidden_dim]
            
            # Apply physical adjacency (vectorized GCN)
            # phys_adj: [17, 17], H_weighted: [valid_batch, 17, hidden_dim]
            H_phys = torch.matmul(phys_adj.unsqueeze(0), H_weighted)  # [valid_batch, 17, hidden_dim]
            
            # Apply edge convolution (fully vectorized)
            Z = self._vectorized_edge_conv(H, scores, combined_adj)  # [valid_batch, 17, edge_hidden]
            
            # Pooling: mean of nodes in this level
            # Apply mask to get only level nodes
            mask = level_mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, H_phys.size(-1))
            
            # Count number of nodes in level for proper mean calculation
            node_count = level_mask.sum().float()
            
            # Masked mean pooling
            H_pooled = (H_phys * mask).sum(dim=1) / node_count  # [valid_batch, hidden_dim]
            Z_pooled = (Z * mask).sum(dim=1) / node_count  # [valid_batch, edge_hidden]
            
            all_H_pooled.append(H_pooled)
            all_Z_pooled.append(Z_pooled)
        
        # Concatenate all pooled features
        all_pooled = torch.cat(all_H_pooled + all_Z_pooled, dim=1)  # [valid_batch, total_dim]
        
        # Apply final projection
        valid_output = self.proj(all_pooled)  # [valid_batch, out_dim]
        
        # Place into output tensor
        output[valid_indices] = valid_output
        
        return output

    def _vectorized_edge_conv(self, H, scores, adjacency):
        """
        Fully vectorized edge convolution with no loops.
        
        Args:
            H: Node features [batch_size, num_nodes, hidden_dim]
            scores: Confidence scores [batch_size, num_nodes, 1]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Edge features [batch_size, num_nodes, edge_hidden]
        """
        batch_size = H.size(0)
        num_nodes = H.size(1)
        hidden_dim = H.size(2)
        device = H.device
        
        # Weight node features by confidence scores
        H_weighted = H * scores  # [batch, num_nodes, hidden_dim]
        
        # Create source and target feature matrices
        # For each node, we need features of all its neighbors
        
        # Step 1: Create source features for each node
        # Expand H_weighted to create source features for each node
        source_feats = H_weighted.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, num_nodes, num_nodes, hidden_dim]
        
        # Step 2: Create target features for each node
        # Expand H_weighted to create target features for each node
        target_feats = H_weighted.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, num_nodes, num_nodes, hidden_dim]
        
        # Step 3: Create edge features by concatenating source and (target - source)
        edge_inputs = torch.cat([
            source_feats,
            target_feats - source_feats
        ], dim=-1)  # [batch, num_nodes, num_nodes, 2*hidden_dim]
        
        # Step 4: Apply edge MLP
        # Reshape for batch processing
        edge_inputs_flat = edge_inputs.view(batch_size * num_nodes * num_nodes, -1)
        edge_outputs_flat = F.relu(self.edge_mlp(edge_inputs_flat))
        edge_outputs = edge_outputs_flat.view(batch_size, num_nodes, num_nodes, -1)  # [batch, num_nodes, num_nodes, edge_hidden]
        
        # Step 5: Apply adjacency mask
        # Expand adjacency to batch dimension and filter connections
        adj_mask = adjacency.unsqueeze(0).unsqueeze(-1)  # [1, num_nodes, num_nodes, 1]
        masked_edge_outputs = edge_outputs * adj_mask  # [batch, num_nodes, num_nodes, edge_hidden]
        
        # Step 6: Max pooling over neighbors for each node
        # First set non-neighbors to -inf (for max pooling)
        adj_mask_bool = (adj_mask > 0).expand_as(masked_edge_outputs)
        masked_edge_outputs = torch.where(adj_mask_bool, masked_edge_outputs, 
                                        torch.tensor(float('-inf'), device=device))
        
        # Max pooling along neighbor dimension
        Z = torch.max(masked_edge_outputs, dim=2)[0]  # [batch, num_nodes, edge_hidden]
        
        # Handle isolated nodes (no neighbors)
        # For nodes with no neighbors, max pooling will return -inf
        # Replace -inf with zeros
        Z = torch.where(Z == float('-inf'), torch.zeros_like(Z), Z)
        
        return Z

class WeightedMLPBasedEncoder(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,          # (x, y) coordinates
        hidden_dims: tuple = (128, 64),
        out_dim: int = 128
    ):
        super().__init__()
        self.num_joints = num_joints

        # MLP layers for encoding keypoints and confidence
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dims[0]),  # Input: (x, y, confidence)
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        # Final projection layer
        self.proj = nn.Linear(hidden_dims[1] * num_joints, out_dim)

    def forward(self, detections: List[Union[dict, None]]) -> torch.Tensor:
        """
        Args:
            detections: list of length N, each either None or a dict:
                {
                  'keypoints': Tensor[17,2],   # (x,y)
                  'scores':    Tensor[17]      # confidences
                }
        Returns:
            Tensor of shape [N, out_dim]
        """
        out_feats = []
        for det in detections:
            if det is None:
                # If no detection, output zero feature
                out_feats.append(torch.zeros(self.proj.out_features))
                continue

            kpts = det['keypoints']      # Tensor[17,2]
            scores = det['scores'].unsqueeze(-1)  # [17,1]

            # Normalize keypoints to [0, 1] based on the bounding box
            min_vals, _ = kpts.min(dim=0, keepdim=True)  # [1, 2]
            max_vals, _ = kpts.max(dim=0, keepdim=True)  # [1, 2]
            kpts = (kpts - min_vals) / (max_vals - min_vals + 1e-6)  # Avoid division by zero

            # Combine keypoints and confidence scores
            P = torch.cat([kpts, scores], dim=-1)  # [17, 3]

            # Pass through MLP
            encoded = self.mlp(P)  # [17, hidden_dims[1]]

            # Weight features by confidence scores
            weighted_encoded = encoded * scores  # [17, hidden_dims[1]]

            # Flatten and project to final output
            flattened = weighted_encoded.view(-1)  # [17 * hidden_dims[1]]
            final_feat = self.proj(flattened)      # [out_dim]
            out_feats.append(final_feat)

        return torch.stack(out_feats, dim=0)  # [N, out_dim]
    

class NaiveMLPEncoder(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,          # (x, y) coordinates
        hidden_dim: int = 128,
        out_dim: int = 128
    ):
        super().__init__()
        self.num_joints = num_joints

        # MLP layers for encoding keypoints and confidence
        self.mlp = nn.Sequential(
            nn.Linear((input_dim + 1) * num_joints, hidden_dim),  # Input: (x, y, confidence) for all joints
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, detections: List[Union[dict, None]]) -> torch.Tensor:
        """
        Args:
            detections: list of length N, each either None or a dict:
                {
                  'keypoints': Tensor[17,2],   # (x,y)
                  'scores':    Tensor[17]      # confidences
                }
        Returns:
            Tensor of shape [N, out_dim]
        """
        out_feats = []
        for det in detections:
            if det is None:
                # If no detection, output zero feature
                out_feats.append(torch.zeros(self.mlp[-1].out_features))
                continue

            kpts = det['keypoints']      # Tensor[17,2]
            scores = det['scores'].unsqueeze(-1)  # [17,1]

            # Normalize keypoints to [0, 1] based on the bounding box
            min_vals, _ = kpts.min(dim=0, keepdim=True)  # [1, 2]
            max_vals, _ = kpts.max(dim=0, keepdim=True)  # [1, 2]
            kpts = (kpts - min_vals) / (max_vals - min_vals + 1e-6)  # Avoid division by zero

            # Combine keypoints and confidence scores
            P = torch.cat([kpts, scores], dim=-1)  # [17, 3]

            # Flatten the input for the MLP
            flattened = P.view(-1)  # [17 * 3]

            # Pass through MLP
            final_feat = self.mlp(flattened)  # [out_dim]
            out_feats.append(final_feat)

        return torch.stack(out_feats, dim=0)  # [N, out_dim]



def count_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The model to count parameters for.
    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Example usage:
if __name__ == "__main__":
    H_encoder = HierarchicalSkeletalEncoder()
    O_encoder = OptimizedHierarchicalEncoder()
    F_encoder = FullyOptimizedHierarchicalEncoder()
    M_encoder = WeightedMLPBasedEncoder()
    N_encoder = NaiveMLPEncoder()
    # Dummy data for 3 frames
    detections = [
        None,
        {
            'keypoints': torch.rand(17, 2)*100,
            'scores':    torch.rand(17)
        },
        {
            'keypoints': torch.rand(17, 2)*100,
            'scores':    torch.rand(17)
        }
    ]
    H_features = H_encoder(detections)
    O_features = O_encoder(detections)
    F_features = F_encoder(detections)
    M_features = M_encoder(detections)
    N_features = N_encoder(detections)
    print("Hierarchical features shape:", H_features.shape)  # should be [3, out_dim]
    print("Optimized features shape:", O_features.shape)  # should be [3, out_dim]
    print("Fully optimized features shape:", F_features.shape)  # should be [3, out_dim]
    print("MLP features shape:", M_features.shape)  # should be [3, out_dim]
    print("Naive MLP features shape:", N_features.shape)  # should be [3, out_dim]
    H_params = count_parameters(H_encoder)
    O_params = count_parameters(O_encoder)
    F_params = count_parameters(F_encoder)
    W_params = count_parameters(M_encoder)
    N_params = count_parameters(N_encoder)

    print(f"HierarchicalSkeletalEncoder parameters: {H_params}")
    print(f"OptimizedHierarchicalEncoder parameters: {O_params}")
    print(f"FullyOptimizedHierarchicalEncoder parameters: {F_params}")
    print(f"WeightedMLPBasedEncoder parameters: {W_params}")
    print(f"NaiveMLPEncoder parameters: {N_params}")