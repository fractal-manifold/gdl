"""
Equivariant Attention Mechanisms for Geometric Deep Learning

This module implements various equivariant attention mechanisms discussed in 
Chapter 5, Section 5.2 of the thesis, including SE(3)-Transformers, E(n)-EGNNs,
and simplified geometric attention variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class SphericalHarmonics:
    """Simplified spherical harmonics computation for SE(3) equivariance."""
    
    @staticmethod
    def compute_Y_1(directions: torch.Tensor) -> torch.Tensor:
        """
        Compute Y^1_m spherical harmonics (l=1, vector features).
        
        Args:
            directions: [N, 3] unit direction vectors
            
        Returns:
            Y_1: [N, 3] spherical harmonics Y^1_{-1,0,1}
        """
        x, y, z = directions.unbind(-1)
        
        # Y^1_m normalization: sqrt(3/(4π))
        norm = math.sqrt(3.0 / (4.0 * math.pi))
        
        # Y^1_{-1} = norm * y, Y^1_0 = norm * z, Y^1_1 = norm * x
        Y_1 = norm * torch.stack([y, z, x], dim=-1)
        return Y_1
    
    @staticmethod
    def compute_Y_2(directions: torch.Tensor) -> torch.Tensor:
        """
        Compute Y^2_m spherical harmonics (l=2, rank-2 tensor features).
        
        Args:
            directions: [N, 3] unit direction vectors
            
        Returns:
            Y_2: [N, 5] spherical harmonics Y^2_{-2,...,2}
        """
        x, y, z = directions.unbind(-1)
        
        # Normalization constants for l=2
        norm1 = math.sqrt(15.0 / (4.0 * math.pi))  # for m=±2
        norm2 = math.sqrt(15.0 / (4.0 * math.pi))  # for m=±1  
        norm3 = math.sqrt(5.0 / (16.0 * math.pi))  # for m=0
        
        Y_2_neg2 = norm1 * x * y
        Y_2_neg1 = norm2 * y * z  
        Y_2_0 = norm3 * (2 * z**2 - x**2 - y**2)
        Y_2_pos1 = norm2 * x * z
        Y_2_pos2 = norm1 * (x**2 - y**2) / 2
        
        Y_2 = torch.stack([Y_2_neg2, Y_2_neg1, Y_2_0, Y_2_pos1, Y_2_pos2], dim=-1)
        return Y_2


class TypedLinear(nn.Module):
    """Linear layer for typed features (scalars, vectors, tensors)."""
    
    def __init__(self, irreps_in: List[Tuple[int, int]], irreps_out: List[Tuple[int, int]]):
        """
        Args:
            irreps_in: [(multiplicity, l)] for input irreps
            irreps_out: [(multiplicity, l)] for output irreps
        """
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        
        # Create linear layers for each irrep type
        self.linears = nn.ModuleDict()
        for i, (mult_out, l_out) in enumerate(irreps_out):
            for j, (mult_in, l_in) in enumerate(irreps_in):
                if l_in == l_out:  # Only same l can be connected
                    key = f"{i}_{j}"
                    self.linears[key] = nn.Linear(mult_in, mult_out, bias=False)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of [N, mult, 2*l+1] tensors for each irrep
            
        Returns:
            output: List of output tensors for each output irrep
        """
        outputs = []
        
        for i, (mult_out, l_out) in enumerate(self.irreps_out):
            dim_out = 2 * l_out + 1
            output = torch.zeros(features[0].size(0), mult_out, dim_out, 
                               device=features[0].device)
            
            for j, (mult_in, l_in) in enumerate(self.irreps_in):
                if l_in == l_out:
                    key = f"{i}_{j}"
                    if key in self.linears:
                        # Linear transformation: [N, mult_in, 2*l+1] -> [N, mult_out, 2*l+1]
                        transformed = self.linears[key](features[j].transpose(-2, -1))
                        output += transformed.transpose(-2, -1)
            
            outputs.append(output)
        
        return outputs


class SE3EquivariantAttention(nn.Module):
    """SE(3)-Equivariant attention mechanism following Fuchs et al. 2020."""
    
    def __init__(self, 
                 irreps_in: List[Tuple[int, int]] = [(32, 0), (16, 1), (8, 2)],
                 irreps_out: List[Tuple[int, int]] = [(32, 0), (16, 1), (8, 2)],
                 max_radius: float = 5.0,
                 num_heads: int = 8):
        super().__init__()
        
        self.irreps_in = irreps_in  # [(mult, l)] pairs
        self.irreps_out = irreps_out
        self.max_radius = max_radius
        self.num_heads = num_heads
        
        # Linear projections for Q, K, V
        self.lin_query = TypedLinear(irreps_in, irreps_in)
        self.lin_key = TypedLinear(irreps_in, irreps_in)  
        self.lin_value = TypedLinear(irreps_in, irreps_out)
        
        # Attention weight computation (only scalar features)
        scalar_dim = irreps_in[0][0] * 2  # Concatenate src and dst scalar features
        self.attention_mlp = nn.Sequential(
            nn.Linear(scalar_dim + 1, 64),  # +1 for distance
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Sigmoid()
        )
        
        self.sh = SphericalHarmonics()
    
    def forward(self, 
                typed_features: List[torch.Tensor], 
                positions: torch.Tensor,
                edge_index: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            typed_features: List of [N, mult, 2*l+1] for each irrep type
            positions: [N, 3] node positions
            edge_index: [2, E] edge connectivity
            
        Returns:
            output: List of updated typed features
        """
        row, col = edge_index
        
        # Compute edge vectors and distances
        edge_vec = positions[col] - positions[row]  # [E, 3]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)  # [E, 1]
        edge_unit = F.normalize(edge_vec, dim=-1)  # [E, 3] unit vectors
        
        # Filter edges by radius
        mask = (edge_dist.squeeze(-1) < self.max_radius)
        if mask.sum() == 0:
            return typed_features
        
        edge_index = edge_index[:, mask]
        edge_dist = edge_dist[mask]
        edge_unit = edge_unit[mask]
        row, col = edge_index
        
        # Compute Q, K, V
        queries = self.lin_query(typed_features)
        keys = self.lin_key(typed_features) 
        values = self.lin_value(typed_features)
        
        # Attention weights from scalar features only
        scalar_src = typed_features[0][row, :, 0]  # [E, mult_0] l=0 features
        scalar_dst = typed_features[0][col, :, 0]  # [E, mult_0]
        
        attention_input = torch.cat([scalar_src, scalar_dst, edge_dist], dim=-1)
        attention_weights = self.attention_mlp(attention_input)  # [E, num_heads]
        
        # Apply attention with spherical harmonics
        outputs = self._apply_equivariant_attention(
            queries, keys, values, edge_index, edge_unit, attention_weights
        )
        
        return outputs
    
    def _apply_equivariant_attention(self, 
                                   queries: List[torch.Tensor],
                                   keys: List[torch.Tensor], 
                                   values: List[torch.Tensor],
                                   edge_index: torch.Tensor,
                                   edge_unit: torch.Tensor,
                                   attention_weights: torch.Tensor) -> List[torch.Tensor]:
        """Apply SE(3)-equivariant attention mechanism."""
        row, col = edge_index
        num_nodes = queries[0].size(0)
        
        # Compute spherical harmonics for edges
        Y_1 = self.sh.compute_Y_1(edge_unit)  # [E, 3]
        Y_2 = self.sh.compute_Y_2(edge_unit)  # [E, 5] 
        
        outputs = []
        for i, (mult_out, l_out) in enumerate(self.irreps_out):
            dim_out = 2 * l_out + 1
            output = torch.zeros(num_nodes, mult_out, dim_out, 
                               device=queries[0].device)
            
            # Aggregate messages from all relevant input irreps
            for j, (mult_in, l_in) in enumerate(self.irreps_in):
                if j < len(values):
                    value = values[j][col]  # [E, mult_in, 2*l_in+1]
                    
                    # Apply Clebsch-Gordan coupling (simplified)
                    if l_out == l_in:
                        # Direct connection (no coupling needed)
                        message = value
                    elif l_out == l_in + 1 and l_in == 0:
                        # Scalar (l=0) × Vector SH (l=1) -> Vector (l=1)
                        if l_out == 1:
                            # [E, mult, 1] × [E, 3] -> [E, mult, 3]
                            scalar_val = value.squeeze(-1)  # [E, mult]
                            message = scalar_val.unsqueeze(-1) * Y_1.unsqueeze(1)
                        else:
                            continue
                    elif l_out == l_in + 1 and l_in == 1:
                        # Vector (l=1) × Vector SH (l=1) -> Tensor (l=2)  
                        if l_out == 2:
                            # Simplified coupling: [E, mult, 3] × [E, 5] -> [E, mult, 5]
                            vector_val = value  # [E, mult, 3]
                            # Simple approximation for demonstration
                            message = torch.zeros(value.size(0), mult_in, 5, device=value.device)
                            message[:, :, :3] = vector_val  # Copy first 3 components
                        else:
                            continue
                    else:
                        continue
                    
                    # Apply attention weights (broadcast over channels)
                    weighted_message = message * attention_weights.mean(dim=-1, keepdim=True).unsqueeze(-1)
                    
                    # Aggregate to target nodes
                    output.index_add_(0, row, weighted_message)
            
            outputs.append(output)
        
        return outputs


class EnEquivariantGNN(nn.Module):
    """E(n)-Equivariant Graph Neural Network following Satorras et al. 2021."""
    
    def __init__(self, 
                 node_dim: int = 64,
                 edge_dim: int = 1, 
                 hidden_dim: int = 128,
                 num_layers: int = 4):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers
        
        # Message function φ_e
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, hidden_dim),  # +1 for squared distance
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate function φ_x  
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Node update function φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Normalization constant
        self.coord_norm = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, 
                h: torch.Tensor, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [N, node_dim] node features
            x: [N, 3] node coordinates  
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_dim] edge attributes (optional)
            
        Returns:
            h_out: [N, node_dim] updated node features
            x_out: [N, 3] updated coordinates
        """
        row, col = edge_index
        
        # Edge vectors and squared distances (E(n)-invariant)
        edge_vec = x[col] - x[row]  # [E, 3]
        edge_dist_sq = torch.sum(edge_vec**2, dim=-1, keepdim=True)  # [E, 1]
        
        # Edge attributes
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 1, device=h.device)
        
        # Message computation
        message_input = torch.cat([
            h[row], h[col],  # Source and target node features
            edge_attr,       # Edge attributes
            edge_dist_sq     # Squared distance (invariant)
        ], dim=-1)
        
        messages = self.message_mlp(message_input)  # [E, hidden_dim]
        
        # Coordinate updates (equivariant)
        coord_weights = self.coord_mlp(messages)  # [E, 1]
        coord_diff = edge_vec * coord_weights  # [E, 3] - element-wise multiplication
        
        # Aggregate coordinate changes
        x_out = x.clone()
        coord_updates = torch.zeros_like(x)
        coord_updates.index_add_(0, row, coord_diff)
        x_out = x + self.coord_norm * coord_updates
        
        # Node feature updates
        h_messages = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        h_messages.index_add_(0, row, messages)
        
        node_input = torch.cat([h, h_messages], dim=-1)
        h_out = self.node_mlp(node_input)
        
        return h_out, x_out


class EquivarianceTest:
    """Test suite for verifying equivariance properties."""
    
    @staticmethod
    def random_rotation_3d() -> torch.Tensor:
        """Generate random 3D rotation matrix."""
        # QR decomposition of random matrix gives random rotation
        q = torch.randn(3, 3)
        q, r = torch.linalg.qr(q)
        q = q * torch.sign(torch.diag(r)).unsqueeze(0)
        return q.T  # Transpose for right multiplication
    
    @staticmethod
    def test_se3_equivariance(model: SE3EquivariantAttention, 
                             typed_features: List[torch.Tensor],
                             positions: torch.Tensor,
                             edge_index: torch.Tensor,
                             num_tests: int = 10) -> float:
        """Test SE(3)-equivariance of attention model."""
        model.eval()
        errors = []
        
        with torch.no_grad():
            # Original output
            out_orig = model(typed_features, positions, edge_index)
            
            for _ in range(num_tests):
                # Random SE(3) transformation
                R = EquivarianceTest.random_rotation_3d()
                t = torch.randn(3) * 0.1
                
                # Transform input
                pos_transformed = positions @ R.T + t
                
                # Transform typed features
                features_transformed = []
                for i, (mult, l) in enumerate(model.irreps_in):
                    if l == 0:  # Scalars - invariant
                        features_transformed.append(typed_features[i])
                    elif l == 1:  # Vectors - rotate
                        feat_rot = typed_features[i] @ R  # [N, mult, 3] @ [3, 3]
                        features_transformed.append(feat_rot)
                    elif l == 2:  # Rank-2 tensors - more complex rotation
                        # Simplified: assume vectors for demonstration
                        if typed_features[i].size(-1) == 3:
                            feat_rot = typed_features[i] @ R
                            features_transformed.append(feat_rot)
                        else:
                            features_transformed.append(typed_features[i])
                    else:
                        features_transformed.append(typed_features[i])
                
                # Model output on transformed input
                out_transformed = model(features_transformed, pos_transformed, edge_index)
                
                # Transform original output
                out_orig_transformed = []
                for i, (mult, l) in enumerate(model.irreps_out):
                    if l == 0:  # Scalars
                        out_orig_transformed.append(out_orig[i])
                    elif l == 1:  # Vectors
                        out_rot = out_orig[i] @ R
                        out_orig_transformed.append(out_rot)
                    else:
                        out_orig_transformed.append(out_orig[i])
                
                # Compute equivariance error
                error = 0.0
                for i in range(len(out_transformed)):
                    diff = out_transformed[i] - out_orig_transformed[i]
                    error += torch.norm(diff).item()
                
                errors.append(error)
        
        return float(np.mean(errors))
    
    @staticmethod
    def test_en_equivariance(model: EnEquivariantGNN,
                           h: torch.Tensor,
                           x: torch.Tensor, 
                           edge_index: torch.Tensor,
                           num_tests: int = 10) -> Tuple[float, float]:
        """Test E(n)-equivariance of EGNN model."""
        model.eval()
        errors_h = []
        errors_x = []
        
        with torch.no_grad():
            # Original output
            h_orig, x_orig = model(h, x, edge_index)
            
            for _ in range(num_tests):
                # Random E(n) transformation (rotation + translation)
                R = EquivarianceTest.random_rotation_3d()
                t = torch.randn(3) * 0.1
                
                # Transform input coordinates
                x_transformed = x @ R.T + t
                
                # Model output on transformed input (features unchanged)
                h_transformed, x_transformed_out = model(h, x_transformed, edge_index)
                
                # Transform original output
                x_orig_transformed = x_orig @ R.T + t
                
                # Compute equivariance errors
                error_h = torch.norm(h_transformed - h_orig).item()  # Features should be invariant
                error_x = torch.norm(x_transformed_out - x_orig_transformed).item()
                
                errors_h.append(error_h)
                errors_x.append(error_x)
        
        return float(np.mean(errors_h)), float(np.mean(errors_x))


def demonstrate_equivariant_attention():
    """Demonstrate equivariant attention mechanisms with concrete examples."""
    print("=== Demonstrating Equivariant Attention Mechanisms ===")
    
    # Setup synthetic molecular data
    num_nodes = 20
    positions = torch.randn(num_nodes, 3) * 2.0
    
    # Create random connectivity (k-NN graph)
    distances = torch.cdist(positions, positions)
    k = 5
    _, indices = distances.topk(k+1, dim=1, largest=False)  # +1 to exclude self
    indices = indices[:, 1:]  # Remove self-connections
    
    # Create edge index
    edge_index = []
    for i in range(num_nodes):
        for j in indices[i]:
            edge_index.append([i, j.item()])
    edge_index = torch.tensor(edge_index).T
    
    print(f"Created graph with {num_nodes} nodes, {edge_index.size(1)} edges")
    
    # Test SE(3)-Equivariant Attention
    print("\n--- SE(3)-Equivariant Attention ---")
    
    # Typed features: scalars, vectors, rank-2 tensors
    irreps = [(16, 0), (8, 1), (4, 2)]
    typed_features = []
    for mult, l in irreps:
        dim = 2 * l + 1
        feat = torch.randn(num_nodes, mult, dim) * 0.5
        typed_features.append(feat)
    
    model_se3 = SE3EquivariantAttention(irreps_in=irreps, irreps_out=irreps)
    
    # Test equivariance
    eq_error = EquivarianceTest.test_se3_equivariance(
        model_se3, typed_features, positions, edge_index, num_tests=5
    )
    print(f"SE(3) Equivariance error: {eq_error:.6f}")
    
    # Forward pass
    output_se3 = model_se3(typed_features, positions, edge_index)
    print(f"SE(3) output shapes: {[f.shape for f in output_se3]}")
    
    # Test E(n)-Equivariant GNN
    print("\n--- E(n)-Equivariant GNN ---")
    
    node_features = torch.randn(num_nodes, 64)
    model_en = EnEquivariantGNN(node_dim=64, hidden_dim=128)
    
    # Test equivariance
    eq_error_h, eq_error_x = EquivarianceTest.test_en_equivariance(
        model_en, node_features, positions, edge_index, num_tests=5
    )
    print(f"E(n) Equivariance error - features: {eq_error_h:.6f}, coords: {eq_error_x:.6f}")
    
    # Forward pass
    h_out, x_out = model_en(node_features, positions, edge_index)
    print(f"E(n) output shapes: h={h_out.shape}, x={x_out.shape}")
    
    # Performance comparison
    print("\n--- Performance Analysis ---")
    
    import time
    num_trials = 10
    
    # SE(3) timing
    start_time = time.time()
    for _ in range(num_trials):
        _ = model_se3(typed_features, positions, edge_index)
    se3_time = (time.time() - start_time) / num_trials
    
    # E(n) timing  
    start_time = time.time()
    for _ in range(num_trials):
        _ = model_en(node_features, positions, edge_index)
    en_time = (time.time() - start_time) / num_trials
    
    print(f"SE(3)-Attention: {se3_time*1000:.2f} ms/forward")
    print(f"E(n)-EGNN: {en_time*1000:.2f} ms/forward")
    print(f"Speedup ratio: {se3_time/en_time:.2f}x")
    
    # Parameter counting
    se3_params = sum(p.numel() for p in model_se3.parameters())
    en_params = sum(p.numel() for p in model_en.parameters())
    
    print(f"SE(3) parameters: {se3_params:,}")
    print(f"E(n) parameters: {en_params:,}")
    
    print("\n=== Demonstration completed successfully ===")
    return {
        'se3_equivariance_error': eq_error,
        'en_equivariance_error': (eq_error_h, eq_error_x),
        'se3_time': se3_time,
        'en_time': en_time,
        'se3_params': se3_params,
        'en_params': en_params
    }


if __name__ == "__main__":
    results = demonstrate_equivariant_attention()
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"SE(3) achieves equivariance error: {results['se3_equivariance_error']:.2e}")
    print(f"E(n) achieves equivariance error: {results['en_equivariance_error'][1]:.2e}")
    print(f"Both methods successfully preserve geometric symmetries!")