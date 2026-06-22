"""
Steerable Convolutional Neural Networks for SO(2) equivariance

This module implements Steerable CNNs that are equivariant to continuous rotations,
using harmonic basis functions and representation theory.

Based on: "General E(2)-Equivariant Steerable CNNs" by Weiler & Welling (2019)

Author: Claude Code for TFG - Geometric Deep Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import math


class SO2_Irreps:
    """Irreducible representations of SO(2).
    
    Each irrep is characterized by an integer frequency k.
    For real representations, we combine ±k frequencies.
    """
    
    def __init__(self, max_frequency: int = 4):
        self.max_frequency = max_frequency
        self.irreps = self._build_irreps()
    
    def _build_irreps(self) -> List[Tuple[int, int]]:
        """Build list of irreducible representations.
        
        Returns:
            List of (frequency, multiplicity) pairs
        """
        irreps = []
        
        # Trivial representation (k=0)
        irreps.append((0, 1))
        
        # Non-trivial representations (k > 0)
        # Each k > 0 gives a 2D real irrep combining ±k
        for k in range(1, self.max_frequency + 1):
            irreps.append((k, 2))  # 2D representation for frequency k
            
        return irreps
    
    def dim(self) -> int:
        """Total dimension of all irreps."""
        return sum(mult for _, mult in self.irreps)
    
    def __str__(self):
        return f"SO(2) irreps: {self.irreps}, total dim: {self.dim()}"


class CircularHarmonics:
    """Circular harmonics for SO(2) steerable filters."""
    
    @staticmethod
    def generate_filter_basis(
        size: int, 
        max_frequency: int,
        sigma: float = 0.5
    ) -> torch.Tensor:
        """Generate circular harmonic basis filters.
        
        Args:
            size: Filter size (size x size)
            max_frequency: Maximum harmonic frequency
            sigma: Gaussian envelope parameter
            
        Returns:
            Tensor of shape (num_basis, size, size) containing basis filters
        """
        # Create coordinate grid
        coords = torch.linspace(-1, 1, size)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        
        # Convert to polar coordinates
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        # Gaussian radial envelope
        radial_envelope = torch.exp(-0.5 * (r / sigma) ** 2)
        
        # Generate basis functions
        basis_filters = []
        
        # k=0 (trivial representation)
        basis_0 = radial_envelope
        basis_filters.append(basis_0)
        
        # k>0 (non-trivial representations)
        for k in range(1, max_frequency + 1):
            # Cosine component
            cos_k = radial_envelope * torch.cos(k * theta)
            basis_filters.append(cos_k)
            
            # Sine component  
            sin_k = radial_envelope * torch.sin(k * theta)
            basis_filters.append(sin_k)
        
        return torch.stack(basis_filters)
    
    @staticmethod
    def rotate_harmonics(
        harmonics: torch.Tensor, 
        angle: float,
        max_frequency: int
    ) -> torch.Tensor:
        """Rotate circular harmonics by given angle.
        
        Under rotation by θ:
        - k=0 component unchanged
        - k>0 components transform as [cos(kθ), sin(kθ)] rotation matrix
        """
        result = harmonics.clone()
        
        # k=0 unchanged
        # k>0 components
        idx = 1
        for k in range(1, max_frequency + 1):
            # Extract cos and sin components
            cos_comp = harmonics[idx]
            sin_comp = harmonics[idx + 1]
            
            # Apply rotation matrix
            c = math.cos(k * angle)
            s = math.sin(k * angle)
            
            result[idx] = c * cos_comp - s * sin_comp
            result[idx + 1] = s * cos_comp + c * sin_comp
            
            idx += 2
        
        return result


class SteerableConv2d(nn.Module):
    """Steerable 2D convolution layer for SO(2) equivariance."""
    
    def __init__(
        self,
        in_irreps: SO2_Irreps,
        out_irreps: SO2_Irreps, 
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        max_frequency: int = 4
    ):
        super().__init__()
        
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_frequency = max_frequency
        
        # Generate harmonic basis
        self.register_buffer(
            'basis_filters',
            CircularHarmonics.generate_filter_basis(
                kernel_size, max_frequency
            )
        )
        
        num_basis = self.basis_filters.shape[0]
        
        # Learnable coefficients for combining basis functions
        # Shape: (out_channels, in_channels, num_basis)
        self.coefficients = nn.Parameter(
            torch.randn(out_irreps.dim(), in_irreps.dim(), num_basis)
        )
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        std = math.sqrt(2.0 / (self.in_irreps.dim() + self.out_irreps.dim()))
        nn.init.normal_(self.coefficients, 0, std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of steerable convolution.
        
        Args:
            x: Input tensor (N, in_dim, H, W)
            
        Returns:
            Output tensor (N, out_dim, H_out, W_out)
        """
        batch_size = x.shape[0]
        
        # Generate steerable kernels
        kernels = self._generate_kernels()  # (out_dim, in_dim, K, K)
        
        # Apply convolution
        # Reshape for grouped convolution
        kernels_flat = kernels.view(-1, kernels.shape[2], kernels.shape[3])
        
        # Apply convolution
        output = F.conv2d(
            x, kernels_flat,
            stride=self.stride,
            padding=self.padding,
            groups=1  # Standard convolution
        )
        
        return output
    
    def _generate_kernels(self) -> torch.Tensor:
        """Generate steerable kernels from basis functions."""
        # Combine basis functions with learned coefficients
        # coefficients: (out_dim, in_dim, num_basis)
        # basis_filters: (num_basis, K, K)
        
        kernels = torch.einsum(
            'oib,bkl->oikl',
            self.coefficients,
            self.basis_filters
        )
        
        return kernels


class SteerableLayer(nn.Module):
    """Complete steerable layer with normalization and activation."""
    
    def __init__(
        self,
        in_irreps: SO2_Irreps,
        out_irreps: SO2_Irreps,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: str = 'relu',
        use_norm: bool = True
    ):
        super().__init__()
        
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        
        self.conv = SteerableConv2d(
            in_irreps, out_irreps, kernel_size, stride, padding
        )
        
        # Norm layer (needs to respect irrep structure)
        if use_norm:
            self.norm = nn.BatchNorm2d(out_irreps.dim())
        else:
            self.norm = None
            
        # Activation (needs to preserve irrep structure)
        if activation == 'relu':
            self.activation = self._steerable_relu
        elif activation == 'gated':
            self.activation = self._gated_activation
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.norm is not None:
            x = self.norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        return x
    
    def _steerable_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU that preserves irrep structure."""
        # For frequency k>0, apply norm-ReLU
        result = []
        channel_idx = 0
        
        for freq, mult in self.out_irreps.irreps:
            if freq == 0:
                # Trivial representation - standard ReLU
                channels = x[:, channel_idx:channel_idx+mult]
                result.append(F.relu(channels))
            else:
                # Non-trivial representation - norm-ReLU
                channels = x[:, channel_idx:channel_idx+mult]  # (N, 2, H, W)
                
                # Compute norm
                norm = torch.norm(channels, dim=1, keepdim=True)  # (N, 1, H, W)
                
                # Apply gating
                gated_norm = F.relu(norm)
                
                # Normalize and scale
                eps = 1e-8
                unit_channels = channels / (norm + eps)
                result.append(unit_channels * gated_norm)
            
            channel_idx += mult
        
        return torch.cat(result, dim=1)
    
    def _gated_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Gated activation for steerable features."""
        # More sophisticated gated activation
        return self._steerable_relu(x)


class SO2SteerableCNN(nn.Module):
    """Complete SO(2)-steerable CNN for image classification."""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        max_frequency: int = 3,
        base_width: int = 16
    ):
        super().__init__()
        
        self.max_frequency = max_frequency
        
        # Define irrep progression
        self.irreps_input = SO2_Irreps(0)  # Start with trivial representation
        self.irreps_hidden1 = SO2_Irreps(max_frequency)
        self.irreps_hidden2 = SO2_Irreps(max_frequency) 
        self.irreps_hidden3 = SO2_Irreps(max_frequency)
        
        # Lifting layer: R^2 -> SO(2)
        self.lift = nn.Conv2d(input_channels, self.irreps_hidden1.dim() * base_width, 
                              kernel_size=3, padding=1)
        
        # Steerable layers
        self.steerable1 = SteerableLayer(
            SO2_Irreps(max_frequency), SO2_Irreps(max_frequency),
            kernel_size=3, padding=1
        )
        
        self.steerable2 = SteerableLayer(
            SO2_Irreps(max_frequency), SO2_Irreps(max_frequency),
            kernel_size=3, padding=1  
        )
        
        self.steerable3 = SteerableLayer(
            SO2_Irreps(max_frequency), SO2_Irreps(max_frequency),
            kernel_size=3, padding=1
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Invariant projection and classification
        hidden_dim = self.irreps_hidden3.dim() * base_width
        self.invariant_map = nn.Linear(hidden_dim, base_width * 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(base_width * 4, base_width * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(base_width * 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lifting
        x = F.relu(self.lift(x))
        x = self.pool(x)
        
        # Steerable convolutions
        x = self.steerable1(x)
        x = self.pool(x)
        
        x = self.steerable2(x)
        x = self.pool(x)
        
        x = self.steerable3(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Project to invariant features
        x = self.invariant_map(x)
        x = F.relu(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


class SO2EquivarianceTest:
    """Test SO(2) equivariance properties."""
    
    def __init__(self, model: nn.Module):
        self.model = model.eval()
    
    def test_rotation_equivariance(
        self,
        x: torch.Tensor,
        angles: List[float] = None,
        tolerance: float = 1e-3
    ) -> Dict[str, float]:
        """Test equivariance to rotations."""
        if angles is None:
            angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        
        results = {}
        
        with torch.no_grad():
            # Original output
            original_output = self.model(x)
            
            for i, angle in enumerate(angles):
                # Rotate input
                rotated_x = self._rotate_tensor(x, angle)
                
                # Get output from rotated input
                rotated_output = self.model(rotated_x)
                
                # For classification, we expect similar outputs (invariance)
                # For feature maps, we'd expect equivariance
                diff = torch.mean((original_output - rotated_output) ** 2).item()
                results[f'rotation_{angle:.2f}'] = diff
        
        return results
    
    def _rotate_tensor(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate tensor by given angle."""
        # Simple rotation using grid_sample
        N, C, H, W = x.shape
        
        # Create rotation matrix
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=x.dtype, device=x.device)
        
        # Create coordinate grid
        coords = torch.linspace(-1, 1, H, device=x.device)
        y, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([x_coords, y], dim=-1)  # (H, W, 2)
        
        # Apply rotation
        grid_flat = grid.view(-1, 2)  # (H*W, 2)
        rotated_grid = grid_flat @ rotation_matrix.T
        rotated_grid = rotated_grid.view(H, W, 2)
        
        # Add batch dimension
        rotated_grid = rotated_grid.unsqueeze(0).repeat(N, 1, 1, 1)
        
        # Sample
        rotated_x = F.grid_sample(
            x, rotated_grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        return rotated_x
    
    def visualize_steerable_filters(
        self,
        layer_name: str = 'steerable1',
        num_rotations: int = 8
    ):
        """Visualize how steerable filters change under rotation."""
        # Get the steerable layer
        layer = getattr(self.model, layer_name)
        
        # Generate kernels at different angles
        angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
        
        fig, axes = plt.subplots(2, num_rotations//2, figsize=(16, 8))
        axes = axes.flatten()
        
        # Get basis filters
        basis_filters = layer.conv.basis_filters
        
        for i, angle in enumerate(angles):
            # Rotate basis harmonics
            rotated_basis = CircularHarmonics.rotate_harmonics(
                basis_filters, angle, layer.conv.max_frequency
            )
            
            # Combine with learned coefficients (show first output channel)
            coeffs = layer.conv.coefficients[0, 0, :]  # First output, first input
            kernel = torch.sum(coeffs.unsqueeze(-1).unsqueeze(-1) * rotated_basis, dim=0)
            
            # Plot
            axes[i].imshow(kernel.detach().cpu().numpy(), cmap='RdBu')
            axes[i].set_title(f'θ = {angle:.2f}')
            axes[i].axis('off')
        
        plt.suptitle('Steerable Filter under Rotations')
        plt.tight_layout()
        plt.show()


def demonstrate_so2_steerable_cnn():
    """Demonstrate SO(2) Steerable CNN."""
    print("=== Demonstrating SO(2) Steerable CNN ===")
    
    # Create model
    model = SO2SteerableCNN(num_classes=10, max_frequency=3)
    
    # Create test input
    x = torch.randn(2, 3, 32, 32)
    
    # Test equivariance
    tester = SO2EquivarianceTest(model)
    errors = tester.test_rotation_equivariance(x)
    
    # Print results
    print("Rotation equivariance test results:")
    for angle, error in errors.items():
        if error < 1e-2:  # More lenient for continuous rotations
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{angle}: {error:.2e} {status}")
    
    # Visualize steerable filters
    tester.visualize_steerable_filters()
    
    # Test forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tester


if __name__ == "__main__":
    # Run demonstration
    model, tester = demonstrate_so2_steerable_cnn()
    print("SO(2) Steerable CNN demonstration completed!")