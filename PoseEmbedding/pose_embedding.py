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
                        edge_msgs.append(torch.zeros(self.edge_mlp.out_features))
                Z = torch.stack(edge_msgs, dim=0)     # [17, edge_hidden]
                Z_levels.append(Z)

            # POOL & PROJECTION
            pooled = []
            for H, Z in zip(H_levels, Z_levels):
                F_i = torch.cat([H, Z], dim=-1)       # [17, hidden_i + edge_hidden]
                pooled.append(F_i.mean(dim=0))        # [hidden_i + edge_hidden]
            final_feat = torch.cat(pooled, dim=-1)    # [sum(hidden_i) + 3*edge_hidden]
            out_feats.append(self.proj(final_feat))   # [out_dim]

        return torch.stack(out_feats, dim=0)         # [N, out_dim]


# Example usage:
if __name__ == "__main__":
    encoder = HierarchicalSkeletalEncoder()
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
    features = encoder(detections)
    print("Output features shape:", features)  # should be [3, out_dim]
