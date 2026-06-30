#!/usr/bin/env python3
"""
Generate CNN kernel visualization figures for the thesis.

This script creates the following figures:
1. Edge detection kernels comparison (Sobel, Prewitt, Laplacian)
2. Gaussian smoothing with different sigma values
3. Sharpening effect before/after
4. CNN kernel to spectral representation on graph grid
5. Gauge transport visualization on cylinder
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.gridspec as gridspec

# Set up matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})


def create_test_image():
    """Create a test image with edges and gradients for demonstration."""
    size = 128
    img = np.zeros((size, size))

    # Add a rectangle
    img[30:90, 30:90] = 1.0

    # Add a circle
    y, x = np.ogrid[:size, :size]
    center = (size // 2 + 20, size // 2 + 20)
    radius = 25
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    img[mask] = 0.7

    # Add a gradient region
    gradient = np.linspace(0, 1, size // 3)
    img[10:25, 10:10 + size // 3] = gradient

    # Add some texture/noise
    np.random.seed(42)
    img += 0.05 * np.random.randn(size, size)
    img = np.clip(img, 0, 1)

    return img


def figure1_edge_detection_kernels():
    """Create figure showing effects of different edge detection kernels."""
    print("Generating Figure 1: Edge detection kernels...")

    # Define kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    img = create_test_image()

    # Apply kernels
    sobel_result = np.sqrt(convolve2d(img, sobel_x, mode='same')**2 +
                          convolve2d(img, sobel_y, mode='same')**2)
    prewitt_result = np.sqrt(convolve2d(img, prewitt_x, mode='same')**2 +
                            convolve2d(img, prewitt_y, mode='same')**2)
    laplacian_result = np.abs(convolve2d(img, laplacian, mode='same'))

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Top row: Original and results
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Imagen original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sobel_result, cmap='gray')
    axes[0, 1].set_title('Sobel (magnitud)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(prewitt_result, cmap='gray')
    axes[0, 2].set_title('Prewitt (magnitud)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(laplacian_result, cmap='gray')
    axes[0, 3].set_title('Laplaciano')
    axes[0, 3].axis('off')

    # Bottom row: Kernel visualizations
    kernels = [sobel_x, sobel_y, prewitt_x, laplacian]
    titles = ['Sobel $G_x$', 'Sobel $G_y$', 'Prewitt $G_x$', 'Laplaciano $L$']

    for i, (kernel, title) in enumerate(zip(kernels, titles)):
        ax = axes[1, i]
        im = ax.imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
        ax.set_title(title)
        for (j, k), val in np.ndenumerate(kernel):
            ax.text(k, j, f'{int(val)}', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('edge_detection_kernels.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: edge_detection_kernels.png")


def figure2_gaussian_smoothing():
    """Create figure comparing different degrees of Gaussian smoothing."""
    print("Generating Figure 2: Gaussian smoothing comparison...")

    img = create_test_image()

    # Apply Gaussian smoothing with different sigma values
    sigmas = [0.5, 1.0, 2.0, 4.0]
    smoothed = [ndimage.gaussian_filter(img, sigma=s) for s in sigmas]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))

    # Original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Smoothed versions
    for i, (s, sm) in enumerate(zip(sigmas, smoothed)):
        axes[i + 1].imshow(sm, cmap='gray')
        axes[i + 1].set_title(f'$\\sigma = {s}$')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('gaussian_smoothing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gaussian_smoothing_comparison.png")


def figure3_sharpening_effect():
    """Create figure showing before/after sharpening effect."""
    print("Generating Figure 3: Sharpening effect...")

    img = create_test_image()

    # Add some blur to make sharpening effect more visible
    img_blurred = ndimage.gaussian_filter(img, sigma=1.0)

    # Sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply sharpening
    img_sharpened = convolve2d(img_blurred, sharpening_kernel, mode='same')
    img_sharpened = np.clip(img_sharpened, 0, 1)

    # Unsharp masking (alternative method)
    blurred_more = ndimage.gaussian_filter(img_blurred, sigma=2.0)
    unsharp_mask = img_blurred - blurred_more
    img_unsharp = img_blurred + 1.5 * unsharp_mask
    img_unsharp = np.clip(img_unsharp, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    axes[0].imshow(img_blurred, cmap='gray')
    axes[0].set_title('Imagen original (ligeramente borrosa)')
    axes[0].axis('off')

    axes[1].imshow(img_sharpened, cmap='gray')
    axes[1].set_title('Después de sharpening ($S = I + L$)')
    axes[1].axis('off')

    # Show the kernel
    ax_kernel = axes[2]
    im = ax_kernel.imshow(sharpening_kernel, cmap='RdBu', vmin=-2, vmax=5)
    ax_kernel.set_title('Kernel de sharpening $S$')
    for (j, k), val in np.ndenumerate(sharpening_kernel):
        ax_kernel.text(k, j, f'{int(val)}', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_kernel.set_xticks([])
    ax_kernel.set_yticks([])

    # Difference image
    diff = np.abs(img_sharpened - img_blurred)
    axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('Diferencia (bordes realzados)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('sharpening_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: sharpening_effect.png")


def figure4_cnn_to_spectral():
    """Create figure showing CNN kernel to spectral representation on graph grid."""
    print("Generating Figure 4: CNN to spectral transformation...")

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.3, 1.2, 1.2])

    # Left: 3x3 CNN kernel on image grid
    ax1 = fig.add_subplot(gs[0])

    # Draw image grid
    grid_size = 5
    for i in range(grid_size + 1):
        ax1.axhline(y=i, color='gray', linewidth=0.5)
        ax1.axvline(x=i, color='gray', linewidth=0.5)

    # Highlight kernel region (center 3x3)
    kernel_colors = np.array([
        [0.2, 0.4, 0.2],
        [0.4, 0.8, 0.4],
        [0.2, 0.4, 0.2]
    ])

    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((j + 1, grid_size - 2 - i), 1, 1,
                                  facecolor=plt.cm.Blues(kernel_colors[i, j]),
                                  edgecolor='blue', linewidth=2)
            ax1.add_patch(rect)
            # Add kernel values
            val = kernel_colors[i, j]
            ax1.text(j + 1.5, grid_size - 1.5 - i, f'{val:.1f}',
                    ha='center', va='center', fontsize=10, fontweight='bold')

    # Mark center pixel
    ax1.plot(2.5, 2.5, 'ro', markersize=10)
    ax1.set_xlim(-0.2, grid_size + 0.2)
    ax1.set_ylim(-0.2, grid_size + 0.2)
    ax1.set_aspect('equal')
    ax1.set_title('Kernel CNN 3×3\nen grilla de imagen')
    ax1.axis('off')

    # Arrow
    ax_arrow = fig.add_subplot(gs[1])
    ax_arrow.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                     arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    ax_arrow.text(0.5, 0.65, 'Transformación\nespectral', ha='center', va='bottom', fontsize=9)
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis('off')

    # Middle: Graph representation
    ax2 = fig.add_subplot(gs[2])

    # Create graph grid vertices
    n = 5
    pos = {}
    for i in range(n):
        for j in range(n):
            pos[(i, j)] = (j, n - 1 - i)

    # Draw edges (4-connectivity)
    for i in range(n):
        for j in range(n):
            if j < n - 1:  # Horizontal edge
                ax2.plot([pos[(i, j)][0], pos[(i, j + 1)][0]],
                        [pos[(i, j)][1], pos[(i, j + 1)][1]],
                        'gray', linewidth=1, zorder=1)
            if i < n - 1:  # Vertical edge
                ax2.plot([pos[(i, j)][0], pos[(i + 1, j)][0]],
                        [pos[(i, j)][1], pos[(i + 1, j)][1]],
                        'gray', linewidth=1, zorder=1)

    # Draw vertices
    for (i, j), (x, y) in pos.items():
        color = 'lightblue'
        size = 300
        if 1 <= i <= 3 and 1 <= j <= 3:
            color = plt.cm.Blues(kernel_colors[i - 1, j - 1])
            size = 400
        ax2.scatter(x, y, s=size, c=[color], edgecolors='black', linewidth=1, zorder=2)

    # Mark center vertex
    ax2.scatter(pos[(2, 2)][0], pos[(2, 2)][1], s=500, c='red',
               edgecolors='darkred', linewidth=2, zorder=3, marker='s')

    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.set_aspect('equal')
    ax2.set_title('Grafo grilla $G = (V, E)$\ncon conectividad 4')
    ax2.axis('off')

    # Right: Spectral filter response
    ax3 = fig.add_subplot(gs[3])

    # Create spectral filter response (Chebyshev approximation style)
    lambdas = np.linspace(0, 2, 100)

    # Example spectral response of a 3x3 averaging kernel
    g_lambda = 1 - 0.3 * lambdas + 0.05 * lambdas**2
    g_lambda = np.clip(g_lambda, 0, 1)

    ax3.plot(lambdas, g_lambda, 'b-', linewidth=2, label='$g(\\lambda)$ original')

    # Chebyshev approximation
    from numpy.polynomial import chebyshev
    # Fit Chebyshev polynomial
    coeffs = np.polyfit(lambdas, g_lambda, 3)
    g_approx = np.polyval(coeffs, lambdas)
    ax3.plot(lambdas, g_approx, 'r--', linewidth=2,
            label='$\\hat{g}(\\lambda) = \\sum_k \\theta_k T_k(\\lambda)$')

    ax3.set_xlabel('Autovalor $\\lambda$')
    ax3.set_ylabel('Respuesta $g(\\lambda)$')
    ax3.set_title('Filtro espectral\n(aprox. Chebyshev)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2)
    ax3.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('cnn_to_spectral_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_to_spectral_graph.png")


def figure5_gauge_transport_cylinder():
    """Create figure showing gauge transport on a cylinder."""
    print("Generating Figure 5: Gauge transport on cylinder...")

    fig = plt.figure(figsize=(14, 5))

    # Create cylinder
    ax1 = fig.add_subplot(131, projection='3d')

    # Cylinder parameters
    theta = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(0, 2, 20)
    Theta, Z = np.meshgrid(theta, z)
    X = np.cos(Theta)
    Y = np.sin(Theta)

    # Plot cylinder surface
    ax1.plot_surface(X, Y, Z, alpha=0.3, color='lightblue', edgecolor='none')

    # Add local coordinate systems at different points
    points = [
        (0, 0),      # theta=0, z=0
        (np.pi/2, 0.5),  # theta=pi/2, z=0.5
        (np.pi, 1.0),    # theta=pi, z=1
        (3*np.pi/2, 1.5) # theta=3pi/2, z=1.5
    ]

    colors = ['red', 'green', 'blue', 'purple']

    for (th, zz), color in zip(points, colors):
        # Point on cylinder
        px, py, pz = np.cos(th), np.sin(th), zz

        # Local tangent vectors (scaled for visibility)
        scale = 0.3
        # Tangent in theta direction
        t_theta = np.array([-np.sin(th), np.cos(th), 0]) * scale
        # Tangent in z direction
        t_z = np.array([0, 0, 1]) * scale

        # Draw coordinate frame
        ax1.quiver(px, py, pz, t_theta[0], t_theta[1], t_theta[2],
                  color=color, arrow_length_ratio=0.2, linewidth=2)
        ax1.quiver(px, py, pz, t_z[0], t_z[1], t_z[2],
                  color=color, arrow_length_ratio=0.2, linewidth=2, linestyle='--')
        ax1.scatter([px], [py], [pz], s=100, c=color, marker='o')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Cilindro con sistemas\nde coordenadas locales')
    ax1.set_box_aspect([1, 1, 1.5])

    # Second plot: Kernel transport
    ax2 = fig.add_subplot(132, projection='3d')

    # Plot cylinder
    ax2.plot_surface(X, Y, Z, alpha=0.2, color='lightblue', edgecolor='none')

    # Draw kernels at different points (represented as small grids)
    def draw_kernel_on_surface(ax, theta, z, color, rotation=0):
        """Draw a 3x3 kernel grid on the cylinder surface."""
        cx, cy, cz = np.cos(theta), np.sin(theta), z

        # Create local coordinate frame
        e_theta = np.array([-np.sin(theta), np.cos(theta), 0])
        e_z = np.array([0, 0, 1])

        # Kernel grid offset
        kernel_size = 0.15
        kernel_values = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.4, 0.2],
            [0.1, 0.2, 0.1]
        ])

        for i in range(3):
            for j in range(3):
                # Position in local coordinates
                local_pos = (i - 1) * kernel_size * e_theta + (j - 1) * kernel_size * e_z
                pos = np.array([cx, cy, cz]) + local_pos

                # Project back to cylinder surface
                r = np.sqrt(pos[0]**2 + pos[1]**2)
                if r > 0:
                    pos[0] /= r
                    pos[1] /= r

                ax.scatter([pos[0]], [pos[1]], [pos[2]],
                          s=100 * kernel_values[i, j] * 3, c=color, alpha=0.7)

    # Draw kernels at different positions
    draw_kernel_on_surface(ax2, 0, 0.5, 'red')
    draw_kernel_on_surface(ax2, np.pi/2, 1.0, 'green')
    draw_kernel_on_surface(ax2, np.pi, 1.5, 'blue')

    # Draw transport paths
    t_path = np.linspace(0, np.pi, 30)
    z_path = np.linspace(0.5, 1.5, 30)
    x_path = np.cos(t_path)
    y_path = np.sin(t_path)
    ax2.plot(x_path, y_path, z_path, 'k--', linewidth=2, alpha=0.7)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Kernels transportados\nentre puntos vecinos')
    ax2.set_box_aspect([1, 1, 1.5])

    # Third plot: Gauge invariance concept
    ax3 = fig.add_subplot(133)

    # Draw schematic of gauge invariance
    # Two different coordinate choices, same pattern

    # Left pattern
    rect1 = plt.Rectangle((0.1, 0.6), 0.3, 0.3, fill=False, edgecolor='blue', linewidth=2)
    ax3.add_patch(rect1)
    ax3.annotate('', xy=(0.35, 0.75), xytext=(0.15, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.annotate('', xy=(0.25, 0.85), xytext=(0.25, 0.65),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(0.25, 0.52, 'Gauge 1', ha='center', fontsize=10)

    # Right pattern (rotated coordinates)
    rect2 = plt.Rectangle((0.6, 0.6), 0.3, 0.3, fill=False, edgecolor='blue', linewidth=2)
    ax3.add_patch(rect2)
    # Rotated arrows
    angle = np.pi/4
    ax3.annotate('', xy=(0.75 + 0.1*np.cos(angle), 0.75 + 0.1*np.sin(angle)),
                xytext=(0.75 - 0.1*np.cos(angle), 0.75 - 0.1*np.sin(angle)),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.annotate('', xy=(0.75 - 0.1*np.sin(angle), 0.75 + 0.1*np.cos(angle)),
                xytext=(0.75 + 0.1*np.sin(angle), 0.75 - 0.1*np.cos(angle)),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(0.75, 0.52, 'Gauge 2', ha='center', fontsize=10)

    # Equality sign
    ax3.text(0.5, 0.75, '=', ha='center', va='center', fontsize=20, fontweight='bold')

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0.2, 1)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Invarianza gauge:\ndiferentes coordenadas,\nmismo resultado')

    plt.tight_layout()
    plt.savefig('gauge_transport_cylinder.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gauge_transport_cylinder.png")


if __name__ == '__main__':
    print("Generating CNN kernel visualization figures...")
    print("=" * 50)

    figure1_edge_detection_kernels()
    figure2_gaussian_smoothing()
    figure3_sharpening_effect()
    figure4_cnn_to_spectral()
    figure5_gauge_transport_cylinder()

    print("=" * 50)
    print("All figures generated successfully!")
