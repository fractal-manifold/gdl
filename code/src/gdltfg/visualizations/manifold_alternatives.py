"""
Generate 3 alternative visualizations for the manifold hypothesis right panel.

Alternatives:
1. Wireframe curved surface with data clusters
2. S-curved ribbon (like a saddle) with points
3. Hemisphere/dome with points distributed on it
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from manifold_figures import generate_torus_points


def create_alternative_1_wireframe(ax):
    """Wireframe curved surface with two data clusters."""
    rng = np.random.default_rng(42)

    # Create surface mesh
    grid_size = 25
    t1 = np.linspace(-2, 2, grid_size)
    t2 = np.linspace(-2, 2, grid_size)
    T1, T2 = np.meshgrid(t1, t2)
    Z = 0.15 * (T1**2 + T2**2) - 0.3

    # Wireframe
    ax.plot_wireframe(T1, T2, Z, alpha=0.4, color='gray', linewidth=0.5, rstride=2, cstride=2)

    # Data clusters ON surface
    n = 150
    # Cluster 1
    x1 = rng.normal(-0.8, 0.35, n)
    y1 = rng.normal(-0.8, 0.35, n)
    z1 = 0.15 * (x1**2 + y1**2) - 0.3

    # Cluster 2
    x2 = rng.normal(0.8, 0.35, n)
    y2 = rng.normal(0.8, 0.35, n)
    z2 = 0.15 * (x2**2 + y2**2) - 0.3

    ax.scatter(x1, y1, z1, c='#E74C3C', s=25, alpha=0.9)
    ax.scatter(x2, y2, z2, c='#3498DB', s=25, alpha=0.9)

    ax.set_box_aspect([1, 1, 0.5])
    ax.view_init(elev=25, azim=45)


def create_alternative_2_saddle(ax):
    """Saddle surface (hyperbolic paraboloid) with data."""
    rng = np.random.default_rng(42)

    # Saddle surface: z = x² - y²
    grid_size = 25
    t1 = np.linspace(-1.5, 1.5, grid_size)
    t2 = np.linspace(-1.5, 1.5, grid_size)
    T1, T2 = np.meshgrid(t1, t2)
    Z = 0.3 * (T1**2 - T2**2)

    # Semi-transparent surface with edges
    ax.plot_surface(T1, T2, Z, alpha=0.3, cmap='coolwarm', edgecolor='gray', linewidth=0.3)

    # Data points on saddle
    n = 200
    x = rng.uniform(-1.2, 1.2, n)
    y = rng.uniform(-1.2, 1.2, n)
    z = 0.3 * (x**2 - y**2)

    # Color by position
    colors = x + y  # gradient across saddle
    ax.scatter(x, y, z, c=colors, cmap='viridis', s=20, alpha=0.9)

    ax.set_box_aspect([1, 1, 0.6])
    ax.view_init(elev=20, azim=35)


def create_alternative_3_hemisphere(ax):
    """Hemisphere/dome with data points on surface."""
    rng = np.random.default_rng(42)

    # Hemisphere surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi / 2, 15)
    U, V = np.meshgrid(u, v)

    R = 1.5
    X = R * np.sin(V) * np.cos(U)
    Y = R * np.sin(V) * np.sin(U)
    Z = R * np.cos(V)

    # Wireframe hemisphere
    ax.plot_wireframe(X, Y, Z, alpha=0.35, color='gray', linewidth=0.4, rstride=1, cstride=2)

    # Data points ON the hemisphere
    n = 300
    # Sample points on hemisphere using rejection sampling
    phi = rng.uniform(0, 2 * np.pi, n)
    theta = np.arccos(rng.uniform(0, 1, n))  # uniform on sphere
    theta = theta[theta < np.pi/2]  # keep only hemisphere
    phi = phi[:len(theta)]

    # Resample to get n points
    while len(theta) < n:
        extra_phi = rng.uniform(0, 2 * np.pi, n)
        extra_theta = np.arccos(rng.uniform(0, 1, n))
        extra_theta = extra_theta[extra_theta < np.pi/2]
        extra_phi = extra_phi[:len(extra_theta)]
        theta = np.concatenate([theta, extra_theta])
        phi = np.concatenate([phi, extra_phi])

    theta = theta[:n]
    phi = phi[:n]

    px = R * np.sin(theta) * np.cos(phi)
    py = R * np.sin(theta) * np.sin(phi)
    pz = R * np.cos(theta)

    ax.scatter(px, py, pz, c=pz, cmap='plasma', s=15, alpha=0.85)

    ax.set_box_aspect([1, 1, 0.7])
    ax.view_init(elev=30, azim=45)


def create_comparison_figure(output_dir: Path):
    """Create all 3 alternatives side by side with the torus."""

    fig = plt.figure(figsize=(16, 4))

    # Torus (reference)
    ax0 = fig.add_subplot(141, projection='3d')
    torus_points, u, v = generate_torus_points(n_points=1000, R=2.5, r=1.0)
    ax0.scatter(torus_points[:, 0], torus_points[:, 1], torus_points[:, 2],
                c=u, cmap='viridis', s=6, alpha=0.8)
    ax0.set_title('Toro T²', fontsize=11)
    ax0.set_box_aspect([1, 1, 0.5])
    ax0.view_init(elev=25, azim=45)
    ax0.grid(True, alpha=0.3)

    # Alternative 1: Wireframe
    ax1 = fig.add_subplot(142, projection='3d')
    create_alternative_1_wireframe(ax1)
    ax1.set_title('Alt 1: Wireframe', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Alternative 2: Saddle
    ax2 = fig.add_subplot(143, projection='3d')
    create_alternative_2_saddle(ax2)
    ax2.set_title('Alt 2: Silla (Saddle)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Alternative 3: Hemisphere
    ax3 = fig.add_subplot(144, projection='3d')
    create_alternative_3_hemisphere(ax3)
    ax3.set_title('Alt 3: Hemisferio', fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "manifold_alternatives_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent.parent
    output_dir = project_root / "src" / "images"
    create_comparison_figure(output_dir)
