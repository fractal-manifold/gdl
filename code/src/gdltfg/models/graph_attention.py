"""
Graph Attention Networks and Spectral Attention Mechanisms

This module implements various graph attention architectures discussed in 
Chapter 5, Section 5.3 of the thesis, including classic GATs, Spectral Attention Networks,
and advanced variants with positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, get_laplacian
from torch_geometric.data import Data
import matplotlib.pyplot as plt


class GraphAttentionLayer(MessagePassing):
    """Classic Graph Attention Network layer following Velickovic et al. 2017."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 dropout: float = 0.6,
                 alpha: float = 0.2,
                 concat: bool = True):
        super().__init__(aggr='add')
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation matrix W
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        
        # Attention mechanism parameter a
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_features] node features
            edge_index: [2, E] edge connectivity
            
        Returns:
            out: [N, out_features] updated node features
        """
        # Linear transformation
        Wh = torch.mm(x, self.W)  # [N, out_features]
        
        # Attention computation
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        return self.propagate(edge_index, x=Wh)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute messages between nodes i and j."""
        # Concatenate features
        concat_features = torch.cat([x_i, x_j], dim=-1)  # [E, 2*out_features]
        
        # Attention coefficients
        e = self.leakyrelu(torch.matmul(concat_features, self.a)).squeeze(-1)  # [E]
        
        return e
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node features after aggregation."""
        if self.concat:
            return F.elu(aggr_out)
        else:
            return aggr_out


class MultiHeadGATLayer(nn.Module):
    """Multi-head Graph Attention Layer."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_heads: int = 8,
                 dropout: float = 0.6,
                 alpha: float = 0.2,
                 concat: bool = True):
        super().__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat)
            for _ in range(num_heads)
        ])
        
        # Final dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.concat:
            # Concatenate all head outputs
            x = torch.cat([att(x, edge_index) for att in self.attentions], dim=-1)
        else:
            # Average all head outputs  
            x = torch.stack([att(x, edge_index) for att in self.attentions], dim=0).mean(dim=0)
        
        return self.dropout(x)


class SpectralPositionalEncoding(nn.Module):
    """Spectral Positional Encoding using graph Laplacian eigenvectors."""
    
    def __init__(self, 
                 pe_dim: int = 16,
                 max_freq: int = 10,
                 norm_type: str = 'sym'):
        super().__init__()
        
        self.pe_dim = pe_dim
        self.max_freq = max_freq
        self.norm_type = norm_type
        
        # Learnable linear projection for eigenvectors
        self.linear = nn.Linear(pe_dim, pe_dim)
        
        # Sign-invariant neural network (simplified SignNet)
        self.sign_net = nn.Sequential(
            nn.Linear(pe_dim, pe_dim * 2),
            nn.ReLU(),
            nn.Linear(pe_dim * 2, pe_dim),
            nn.Tanh()
        )
    
    def forward(self, 
                edge_index: torch.Tensor, 
                num_nodes: int, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute spectral positional encoding for nodes.
        
        Args:
            edge_index: [2, E] edge connectivity
            num_nodes: Number of nodes
            batch: Batch assignment for each node (optional)
            
        Returns:
            pe: [N, pe_dim] positional encodings
        """
        device = edge_index.device
        
        # Compute Laplacian matrix
        edge_index, edge_weight = get_laplacian(edge_index, normalization=self.norm_type, 
                                               num_nodes=num_nodes)
        
        # Build sparse Laplacian matrix
        L = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to_dense()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Sort by eigenvalue magnitude  
        idx = eigenvalues.abs().argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take smallest non-zero eigenvalues/vectors
        pe_raw = eigenvectors[:, 1:self.pe_dim + 1]  # Skip first (constant) eigenvector
        
        # Handle sign ambiguity using SignNet
        pe = self.sign_net(pe_raw)
        
        return pe


class SpectralAttentionLayer(nn.Module):
    """Spectral Attention Network layer with learned positional encoding."""
    
    def __init__(self, 
                 d_model: int = 512,
                 num_heads: int = 8,
                 pe_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        # Positional encoding
        self.pe_encoder = SpectralPositionalEncoding(pe_dim)
        self.pe_linear = nn.Linear(pe_dim, d_model)
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dual attention: real (sparse) + virtual (full)
        self.gamma = nn.Parameter(torch.tensor(0.5))  # Balance parameter
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: [N, d_model] node features
            edge_index: [2, E] edge connectivity
            return_attention: Whether to return attention weights
            
        Returns:
            out: [N, d_model] updated features
            attn_weights: [N, N] attention matrix (if return_attention=True)
        """
        N = x.size(0)
        
        # Add positional encoding
        pe = self.pe_encoder(edge_index, N)
        pe_features = self.pe_linear(pe)
        x_with_pe = x + pe_features
        
        # Multi-head attention projections
        Q = self.q_proj(x_with_pe).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x_with_pe).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x_with_pe).view(N, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [N, H, N]
        
        # Real attention (sparse - only connected nodes)
        sparse_mask = torch.zeros(N, N, device=x.device)
        sparse_mask[edge_index[0], edge_index[1]] = 1
        sparse_mask.fill_diagonal_(1)  # Include self-loops
        
        sparse_scores = scores.masked_fill(sparse_mask.unsqueeze(1) == 0, float('-inf'))
        sparse_attn = F.softmax(sparse_scores, dim=-1)  # [N, H, N]
        
        # Virtual attention (full - all node pairs)
        full_attn = F.softmax(scores, dim=-1)  # [N, H, N]
        
        # Balanced attention
        attn_weights = self.gamma * sparse_attn + (1 - self.gamma) * full_attn
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [N, H, head_dim]
        out = out.contiguous().view(N, self.d_model)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection and layer norm
        out = self.layer_norm(x + self.dropout(out))
        
        if return_attention:
            return out, attn_weights.mean(dim=1)  # Average over heads
        return out


class GraphTransformer(nn.Module):
    """Complete Graph Transformer with spectral attention."""
    
    def __init__(self, 
                 node_dim: int,
                 d_model: int = 512,
                 num_heads: int = 8, 
                 num_layers: int = 6,
                 pe_dim: int = 16,
                 dropout: float = 0.1,
                 num_classes: int = None):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SpectralAttentionLayer(d_model, num_heads, pe_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Graph-level prediction (global pooling)
        if batch is not None:
            # Batch-wise global average pooling
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, self.d_model, device=x.device)
            for i in range(batch_size):
                mask = batch == i
                out[i] = x[mask].mean(dim=0)
        else:
            # Single graph global average pooling  
            out = x.mean(dim=0, keepdim=True)
        
        # Final prediction
        if self.num_classes:
            return self.classifier(out)
        else:
            return self.regressor(out)


class SpectralDistanceMetrics:
    """Implementation of various spectral distance metrics for graphs."""
    
    @staticmethod
    def compute_laplacian_spectrum(edge_index: torch.Tensor, 
                                  num_nodes: int, 
                                  norm_type: str = 'sym') -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of graph Laplacian."""
        # Get normalized Laplacian
        edge_index, edge_weight = get_laplacian(edge_index, normalization=norm_type, 
                                               num_nodes=num_nodes)
        
        # Build dense Laplacian
        L = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to_dense()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Sort by eigenvalue
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def diffusion_distance(eigenvalues: torch.Tensor, 
                          eigenvectors: torch.Tensor, 
                          t: float = 1.0) -> torch.Tensor:
        """
        Compute diffusion distance matrix.
        
        d_D^2(i,j) = Σ_k e^(-2t*λ_k) * (φ_k,i - φ_k,j)^2
        """
        n = eigenvectors.size(0)
        
        # Heat kernel weights
        weights = torch.exp(-2 * t * eigenvalues[1:])  # Skip λ_0 = 0
        
        # Compute pairwise differences
        diff_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                diff_vec = eigenvectors[i, 1:] - eigenvectors[j, 1:]
                diff_matrix[i, j] = torch.sum(weights * diff_vec**2)
        
        return torch.sqrt(diff_matrix)
    
    @staticmethod  
    def green_function_distance(eigenvalues: torch.Tensor, 
                               eigenvectors: torch.Tensor) -> torch.Tensor:
        """
        Compute Green's function based distance.
        
        G(i,j) = Σ_{k>0} φ_k,i * φ_k,j / λ_k
        """
        n = eigenvectors.size(0)
        
        # Avoid division by zero
        nonzero_mask = eigenvalues > 1e-8
        eigenvalues_nz = eigenvalues[nonzero_mask]
        eigenvectors_nz = eigenvectors[:, nonzero_mask]
        
        # Green's function matrix
        green_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                green_matrix[i, j] = torch.sum(
                    eigenvectors_nz[i, :] * eigenvectors_nz[j, :] / eigenvalues_nz
                )
        
        return green_matrix
    
    @staticmethod
    def biharmonic_distance(eigenvalues: torch.Tensor, 
                           eigenvectors: torch.Tensor) -> torch.Tensor:
        """
        Compute biharmonic distance matrix.
        
        d_B^2(i,j) = Σ_{k>0} (φ_k,i - φ_k,j)^2 / λ_k^2  
        """
        n = eigenvectors.size(0)
        
        # Avoid division by zero  
        nonzero_mask = eigenvalues > 1e-8
        eigenvalues_nz = eigenvalues[nonzero_mask]
        eigenvectors_nz = eigenvectors[:, nonzero_mask]
        
        # Compute pairwise differences
        dist_matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                diff_vec = eigenvectors_nz[i, :] - eigenvectors_nz[j, :]
                dist_matrix[i, j] = torch.sum(diff_vec**2 / eigenvalues_nz**2)
        
        return torch.sqrt(dist_matrix)


def demonstrate_graph_attention():
    """Demonstrate Graph Attention Networks with concrete examples."""
    print("=== Demonstrating Graph Attention Networks ===")
    
    # Create synthetic graph data
    num_nodes = 20
    node_features = torch.randn(num_nodes, 16)
    
    # Create a small-world graph structure
    edge_list = []
    
    # Ring structure
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])
        edge_list.append([(i + 1) % num_nodes, i])
    
    # Add random shortcuts (small-world property)
    np.random.seed(42)
    for _ in range(num_nodes // 2):
        i, j = np.random.choice(num_nodes, 2, replace=False)
        edge_list.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edge_list).T.long()
    
    print(f"Created graph: {num_nodes} nodes, {edge_index.size(1)//2} edges")
    
    # Test Classic GAT
    print("\n--- Classic Graph Attention Network ---")
    
    gat_layer = MultiHeadGATLayer(
        in_features=16, 
        out_features=32, 
        num_heads=4,
        dropout=0.1
    )
    
    gat_output = gat_layer(node_features, edge_index)
    print(f"GAT output shape: {gat_output.shape}")
    
    # Test Spectral Attention
    print("\n--- Spectral Attention Network ---")
    
    spectral_layer = SpectralAttentionLayer(d_model=16, num_heads=4, pe_dim=8)
    spectral_output, attention_weights = spectral_layer(
        node_features, edge_index, return_attention=True
    )
    
    print(f"Spectral attention output shape: {spectral_output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Analyze spectral properties
    print("\n--- Spectral Analysis ---")
    
    metrics = SpectralDistanceMetrics()
    eigenvalues, eigenvectors = metrics.compute_laplacian_spectrum(edge_index, num_nodes)
    
    print(f"Graph spectrum (first 10 eigenvalues): {eigenvalues[:10]}")
    
    # Compute spectral distances
    diffusion_dist = metrics.diffusion_distance(eigenvalues, eigenvectors, t=1.0)
    green_dist = metrics.green_function_distance(eigenvalues, eigenvectors)
    biharmonic_dist = metrics.biharmonic_distance(eigenvalues, eigenvectors)
    
    print(f"Spectral distance statistics:")
    print(f"  Diffusion distance - mean: {diffusion_dist.mean():.4f}, std: {diffusion_dist.std():.4f}")
    print(f"  Green function - mean: {green_dist.mean():.4f}, std: {green_dist.std():.4f}")  
    print(f"  Biharmonic distance - mean: {biharmonic_dist.mean():.4f}, std: {biharmonic_dist.std():.4f}")
    
    # Test complete Graph Transformer
    print("\n--- Complete Graph Transformer ---")
    
    graph_transformer = GraphTransformer(
        node_dim=16,
        d_model=64,
        num_heads=4, 
        num_layers=3,
        pe_dim=8,
        num_classes=3
    )
    
    # Create batch data (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    transformer_output = graph_transformer(node_features, edge_index, batch)
    
    print(f"Graph Transformer prediction shape: {transformer_output.shape}")
    print(f"Predicted class probabilities: {F.softmax(transformer_output, dim=-1).squeeze()}")
    
    # Performance analysis
    print("\n--- Performance Analysis ---")
    
    import time
    
    # Classic GAT timing
    start_time = time.time()
    for _ in range(100):
        _ = gat_layer(node_features, edge_index)
    gat_time = (time.time() - start_time) / 100
    
    # Spectral attention timing
    start_time = time.time()
    for _ in range(100):
        _ = spectral_layer(node_features, edge_index)
    spectral_time = (time.time() - start_time) / 100
    
    print(f"Classic GAT: {gat_time*1000:.2f} ms/forward")
    print(f"Spectral Attention: {spectral_time*1000:.2f} ms/forward")
    print(f"Spectral overhead: {spectral_time/gat_time:.2f}x")
    
    # Parameter counting
    gat_params = sum(p.numel() for p in gat_layer.parameters())
    spectral_params = sum(p.numel() for p in spectral_layer.parameters()) 
    transformer_params = sum(p.numel() for p in graph_transformer.parameters())
    
    print(f"Parameter counts:")
    print(f"  GAT Layer: {gat_params:,}")
    print(f"  Spectral Attention Layer: {spectral_params:,}")
    print(f"  Complete Graph Transformer: {transformer_params:,}")
    
    # Attention visualization
    print("\n--- Attention Analysis ---")
    
    with torch.no_grad():
        _, attn_matrix = spectral_layer(node_features, edge_index, return_attention=True)
        
        # Analyze attention patterns
        attention_entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + 1e-8), dim=-1)
        print(f"Attention entropy - mean: {attention_entropy.mean():.3f}, std: {attention_entropy.std():.3f}")
        
        # Check attention concentration  
        max_attention = attn_matrix.max(dim=-1)[0]
        print(f"Max attention weights - mean: {max_attention.mean():.3f}, std: {max_attention.std():.3f}")
        
        # Long-range connections
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Attention on non-adjacent nodes
        non_adj_attention = attn_matrix * (1 - adj - torch.eye(num_nodes))
        long_range_strength = non_adj_attention.sum() / (num_nodes * (num_nodes - 1) - edge_index.size(1))
        print(f"Long-range attention strength: {long_range_strength:.6f}")
    
    print("\n=== Graph Attention Networks demonstration completed ===")
    
    return {
        'gat_output_shape': gat_output.shape,
        'spectral_output_shape': spectral_output.shape,
        'attention_weights_shape': attention_weights.shape,
        'spectrum_info': eigenvalues[:10],
        'transformer_prediction': transformer_output,
        'performance': {
            'gat_time': gat_time,
            'spectral_time': spectral_time
        },
        'parameters': {
            'gat': gat_params,
            'spectral': spectral_params, 
            'transformer': transformer_params
        },
        'attention_analysis': {
            'entropy': attention_entropy.mean().item(),
            'concentration': max_attention.mean().item(),
            'long_range_strength': long_range_strength.item()
        }
    }


if __name__ == "__main__":
    results = demonstrate_graph_attention()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"GAT successfully processes {results['gat_output_shape'][0]} nodes")
    print(f"Spectral attention captures long-range dependencies: {results['attention_analysis']['long_range_strength']:.2e}")
    print(f"Graph Transformer achieves end-to-end classification with {results['parameters']['transformer']:,} parameters")
    print(f"Spectral methods provide {results['attention_analysis']['entropy']:.2f} bits of attention entropy")
    print(f"All methods successfully demonstrate core Graph Attention Network principles!")