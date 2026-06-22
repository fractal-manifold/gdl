"""
Generate manifold hypothesis illustration figure.

This script creates a two-panel visualization:
- Left: Torus T² as a 2D manifold embedded in R³
- Right: High-dimensional data concentrated near a low-dimensional manifold

The torus is a classic example in differential geometry with interesting
topological properties (genus 1, non-trivial fundamental group).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def generate_torus_points(
    n_points: int = 800,
    R: float = 3.0,  # Major radius (center of tube to center of torus)
    r: float = 1.0,  # Minor radius (radius of tube)
    noise: float = 0.0,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points on a torus surface.

    The torus is parametrized by:
        x = (R + r*cos(v)) * cos(u)
        y = (R + r*cos(v)) * sin(u)
        z = r * sin(v)

    where u, v ∈ [0, 2π).

    Parameters
    ----------
    n_points : int
        Number of points to generate
    R : float
        Major radius (distance from center of torus to center of tube)
    r : float
        Minor radius (radius of the tube)
    noise : float
        Standard deviation of Gaussian noise to add
    seed : int
        Random seed for reproducibility

    Returns
    -------
    points : ndarray of shape (n_points, 3)
        Points on the torus
    u : ndarray
        Angular parameter u for coloring
    v : ndarray
        Angular parameter v for coloring
    """
    rng = np.random.default_rng(seed)

    # Sample angles uniformly
    u = rng.uniform(0, 2 * np.pi, n_points)
    v = rng.uniform(0, 2 * np.pi, n_points)

    # Parametric equations for torus
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    # Add noise if requested
    if noise > 0:
        x += rng.normal(0, noise, n_points)
        y += rng.normal(0, noise, n_points)
        z += rng.normal(0, noise, n_points)

    points = np.column_stack([x, y, z])
    return points, u, v


def generate_saddle_manifold(
    n_points: int = 200,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data on a saddle surface (hyperbolic paraboloid).

    The saddle z = x² - y² has negative Gaussian curvature,
    making it a classic example in differential geometry.

    Returns
    -------
    points : ndarray of shape (n_points, 3)
        Points on the saddle surface
    colors : ndarray
        Color values for gradient coloring
    surface_X, surface_Y, surface_Z : ndarrays
        Mesh grid for plotting the underlying surface
    """
    rng = np.random.default_rng(seed)

    # Create the saddle surface mesh: z = x² - y²
    grid_size = 25
    t1_grid = np.linspace(-1.5, 1.5, grid_size)
    t2_grid = np.linspace(-1.5, 1.5, grid_size)
    T1, T2 = np.meshgrid(t1_grid, t2_grid)

    # Saddle surface (hyperbolic paraboloid)
    surface_Z = 0.3 * (T1**2 - T2**2)

    # Generate data points uniformly ON the saddle surface
    x = rng.uniform(-1.2, 1.2, n_points)
    y = rng.uniform(-1.2, 1.2, n_points)
    z = 0.3 * (x**2 - y**2)

    points = np.column_stack([x, y, z])

    # Color by position (x + y gives diagonal gradient)
    colors = x + y

    return points, colors, T1, T2, surface_Z


def create_manifold_hypothesis_figure(
    output_path: str | Path,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 150
) -> None:
    """
    Create the manifold hypothesis illustration figure.

    Parameters
    ----------
    output_path : str or Path
        Path to save the output figure
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Resolution in dots per inch
    """
    fig = plt.figure(figsize=figsize)

    # Left panel: Torus T²
    ax1 = fig.add_subplot(121, projection='3d')

    torus_points, u, v = generate_torus_points(n_points=1000, R=2.5, r=1.0)

    # Color by the u parameter (angle around the torus)
    scatter1 = ax1.scatter(
        torus_points[:, 0],
        torus_points[:, 1],
        torus_points[:, 2],
        c=u,
        cmap='viridis',
        s=8,
        alpha=0.8
    )

    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_zlabel('z', fontsize=10)
    ax1.set_box_aspect([1, 1, 0.5])
    ax1.view_init(elev=25, azim=45)

    # Remove grid for cleaner look
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Right panel: Saddle surface with data points (viridis colormap)
    ax2 = fig.add_subplot(122, projection='3d')

    points, colors, surf_X, surf_Y, surf_Z = generate_saddle_manifold(n_points=250)

    # Compute surface colors to match point colormap
    surf_colors = surf_X + surf_Y  # Same coloring scheme as points

    # Plot the saddle surface with viridis colormap
    ax2.plot_surface(
        surf_X, surf_Y, surf_Z,
        facecolors=plt.cm.viridis((surf_colors - surf_colors.min()) / (surf_colors.max() - surf_colors.min())),
        alpha=0.35,
        edgecolor='gray',
        linewidth=0.2
    )

    # Plot data points ON the surface with matching viridis colors
    ax2.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap='viridis',
        s=20,
        alpha=0.9
    )

    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    ax2.set_zlabel('z', fontsize=10)
    ax2.set_box_aspect([1, 1, 0.6])
    ax2.view_init(elev=20, azim=35)

    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    # Default output path for the thesis
    project_root = Path(__file__).parent.parent.parent.parent.parent
    output_path = project_root / "src" / "images" / "manifold_hypothesis.png"

    create_manifold_hypothesis_figure(output_path)
