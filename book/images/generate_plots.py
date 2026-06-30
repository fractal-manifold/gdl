#!/usr/bin/env python3
"""Generate plots for the TFG thesis: activation functions and loss landscapes."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Avoid math text rendering issues
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_fn(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

def leaky_relu(z, alpha=0.1):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.1):
    return np.where(z > 0, 1.0, alpha)


def plot_activation_functions():
    """Generate A3: Activation functions and their derivatives."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    z = np.linspace(-5, 5, 500)

    functions = [
        ('Sigmoide', sigmoid, sigmoid_derivative, 'blue'),
        ('Tanh', tanh_fn, tanh_derivative, 'green'),
        ('ReLU', relu, relu_derivative, 'red'),
        ('Leaky ReLU', leaky_relu, leaky_relu_derivative, 'purple'),
    ]

    for ax, (name, func, deriv, color) in zip(axes.flat, functions):
        # Plot function
        ax.plot(z, func(z), color=color, linewidth=2, label='f(z)')
        # Plot derivative
        ax.plot(z, deriv(z), color=color, linewidth=2, linestyle='--',
                label="f'(z)", alpha=0.7)

        ax.set_title(name, fontweight='bold', fontsize=12)
        ax.set_xlabel('z', fontsize=10)
        ax.set_ylabel('Valor', fontsize=10)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(-5, 5)

        # Special annotations
        if name == 'Sigmoide':
            ax.annotate('Saturacion', xy=(3.5, 0.95), fontsize=8, color='gray')
            ax.axhspan(0.9, 1.0, alpha=0.1, color='red')
            ax.axhspan(0.0, 0.1, alpha=0.1, color='red')
        elif name == 'Tanh':
            ax.annotate('Saturacion', xy=(3, 0.9), fontsize=8, color='gray')
            ax.axhspan(0.9, 1.0, alpha=0.1, color='red')
            ax.axhspan(-1.0, -0.9, alpha=0.1, color='red')
        elif name == 'ReLU':
            ax.annotate('Zona muerta\n(gradiente=0)', xy=(-3, -0.3), fontsize=8,
                       color='gray', ha='center')
            ax.axvspan(-5, 0, alpha=0.1, color='red')

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: activation_functions.png")


def plot_loss_landscapes():
    """Generate A10: Loss landscapes (convex vs non-convex)."""

    fig = plt.figure(figsize=(12, 5))

    # Create mesh
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # --- Convex loss (paraboloid) ---
    ax1 = fig.add_subplot(121, projection='3d')
    Z_convex = X**2 + Y**2
    surf1 = ax1.plot_surface(X, Y, Z_convex, cmap='viridis', alpha=0.8,
                              linewidth=0, antialiased=True)
    ax1.set_title('Superficie de perdida convexa', fontweight='bold', fontsize=11)
    ax1.set_xlabel('theta_1', fontsize=9)
    ax1.set_ylabel('theta_2', fontsize=9)
    ax1.set_zlabel('L(theta)', fontsize=9)
    ax1.view_init(elev=25, azim=45)

    # Mark global minimum
    ax1.scatter([0], [0], [0], color='red', s=100, marker='*',
                label='Minimo global unico')
    ax1.legend(loc='upper right', fontsize=8)

    # --- Non-convex loss (typical in deep learning) ---
    ax2 = fig.add_subplot(122, projection='3d')

    # Rastrigin-like function (multiple local minima)
    A = 2
    Z_nonconvex = (X**2 + Y**2) + A * (2 - np.cos(2*np.pi*X) - np.cos(2*np.pi*Y))

    surf2 = ax2.plot_surface(X, Y, Z_nonconvex, cmap='plasma', alpha=0.8,
                              linewidth=0, antialiased=True)
    ax2.set_title('Superficie de perdida no convexa\n(tipica en Deep Learning)',
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel('theta_1', fontsize=9)
    ax2.set_ylabel('theta_2', fontsize=9)
    ax2.set_zlabel('L(theta)', fontsize=9)
    ax2.view_init(elev=25, azim=45)

    # Mark some local minima
    local_mins = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    for xm, ym in local_mins:
        zm = (xm**2 + ym**2) + A * (2 - np.cos(2*np.pi*xm) - np.cos(2*np.pi*ym))
        ax2.scatter([xm], [ym], [zm], color='red', s=50, marker='o')

    ax2.scatter([], [], [], color='red', s=50, marker='o',
                label='Minimos locales')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('loss_landscapes.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: loss_landscapes.png")


if __name__ == '__main__':
    print("Generating plots...")
    plot_activation_functions()
    plot_loss_landscapes()
    print("\nAll plots generated successfully!")
