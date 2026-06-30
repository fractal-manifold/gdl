#!/usr/bin/env python3
"""
Generación de figuras para el capítulo GDL
Ejecutar: python generate_gdl_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import os

# Configuración global - usar backend Agg y evitar problemas de fuentes
import matplotlib
matplotlib.use('Agg')

# Reconstruir caché de fuentes y usar fuentes disponibles
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

# Configuración de plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['text.usetex'] = False
# Usar fuente que existe en el sistema
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_graph_laplacian_eigenvectors():
    """Autovectores del Laplaciano: frecuencias bajas vs altas"""
    # Crear un grafo de grilla 8x8
    n = 8
    G = nx.grid_2d_graph(n, n)
    pos = {(i, j): (i, j) for i in range(n) for j in range(n)}

    # Calcular Laplaciano
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = eigh(L)

    # Seleccionar autovectores representativos
    indices = [1, 2, 3, len(eigenvalues)-3, len(eigenvalues)-2, len(eigenvalues)-1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (ax, ev_idx) in enumerate(zip(axes, indices)):
        ev = eigenvectors[:, ev_idx]
        node_colors = ev

        # Normalizar colores
        vmin, vmax = ev.min(), ev.max()

        # Crear posiciones como array
        node_list = list(G.nodes())
        pos_array = np.array([pos[node] for node in node_list])

        # Dibujar
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')
        sc = ax.scatter(pos_array[:, 0], pos_array[:, 1],
                       c=node_colors, cmap='RdBu_r',
                       s=100, vmin=-abs(node_colors).max(), vmax=abs(node_colors).max())

        ax.set_title(f'lambda_{ev_idx+1} = {eigenvalues[ev_idx]:.2f}')
        ax.set_aspect('equal')
        ax.axis('off')

        # Etiqueta de frecuencia
        if idx < 3:
            ax.text(0.5, -0.1, 'Baja frecuencia', transform=ax.transAxes,
                   ha='center', fontsize=10, color='blue')
        else:
            ax.text(0.5, -0.1, 'Alta frecuencia', transform=ax.transAxes,
                   ha='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'laplacian_eigenvectors.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ laplacian_eigenvectors.png generado")


def plot_eigenvalue_spectrum():
    """Espectro de autovalores del Laplaciano"""
    # Comparar diferentes grafos
    n = 50

    graphs = {
        'Camino': nx.path_graph(n),
        'Ciclo': nx.cycle_graph(n),
        'Grilla 7×7': nx.grid_2d_graph(7, 7),
        'Aleatorio (Erdős-Rényi)': nx.erdos_renyi_graph(n, 0.1, seed=42)
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for ax, (name, G), color in zip(axes, graphs.items(), colors):
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(L)

        ax.bar(range(len(eigenvalues)), sorted(eigenvalues),
               color=color, alpha=0.7, width=1.0)
        ax.set_xlabel('Indice i')
        ax.set_ylabel('lambda_i')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Espectro del Laplaciano para Diferentes Grafos', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eigenvalue_spectrum.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ eigenvalue_spectrum.png generado")


def plot_graph_fourier_transform():
    """Transformada de Fourier en grafos"""
    # Grafo pequeño para visualización
    n = 6
    G = nx.cycle_graph(n)
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues, U = eigh(L)

    # Señal de ejemplo
    signal = np.array([1, 0.5, 0, -0.5, -1, 0])

    # Transformada de Fourier
    signal_hat = U.T @ signal

    # Reconstrucción parcial
    k_values = [2, 4, 6]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    pos = nx.circular_layout(G)

    # Fila 1: Señal original y coeficientes
    ax = axes[0, 0]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=signal, cmap='RdBu_r',
                           node_size=500, ax=ax, vmin=-1, vmax=1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title('Senal original f')
    ax.axis('off')

    ax = axes[0, 1]
    ax.bar(range(n), signal_hat, color='purple', alpha=0.7)
    ax.set_xlabel('Frecuencia k')
    ax.set_ylabel('f_k')
    ax.set_title('Coeficientes de Fourier')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.bar(range(n), eigenvalues, color='teal', alpha=0.7)
    ax.set_xlabel('Indice k')
    ax.set_ylabel('lambda_k')
    ax.set_title('Espectro (frecuencias)')
    ax.grid(True, alpha=0.3)

    # Fila 2: Reconstrucciones parciales
    for idx, k in enumerate(k_values):
        ax = axes[1, idx]
        # Reconstruir con primeros k componentes
        signal_rec = U[:, :k] @ signal_hat[:k]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, node_color=signal_rec, cmap='RdBu_r',
                               node_size=500, ax=ax, vmin=-1, vmax=1)
        ax.set_title(f'Reconstrucción ($k={k}$ modos)')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'graph_fourier_transform.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ graph_fourier_transform.png generado")


def _poincare_geodesic_arc(p1, p2, n_pts=80):
    """Compute the geodesic (arc of circle orthogonal to the unit circle)
    between two points p1, p2 inside the Poincaré disk.
    Returns arrays (xs, ys) for plotting."""
    x1, y1 = p1
    x2, y2 = p2
    # If points are nearly collinear through the origin, geodesic is a straight line
    cross = x1 * y2 - x2 * y1
    if abs(cross) < 1e-8:
        return np.linspace(x1, x2, n_pts), np.linspace(y1, y2, n_pts)

    # Inverse images w.r.t. unit circle: p* = p / |p|^2
    r1sq = x1**2 + y1**2
    r2sq = x2**2 + y2**2
    if r1sq < 1e-12 or r2sq < 1e-12:
        return np.linspace(x1, x2, n_pts), np.linspace(y1, y2, n_pts)

    x1s, y1s = x1 / r1sq, y1 / r1sq
    x2s, y2s = x2 / r2sq, y2 / r2sq

    # Centre of the circle through p1, p2, p1*, p2* (any 3 of them)
    # Using perpendicular bisectors of (p1, p1*) and (p2, p2*)
    # Midpoints
    mx1, my1 = (x1 + x1s) / 2, (y1 + y1s) / 2
    mx2, my2 = (x2 + x2s) / 2, (y2 + y2s) / 2
    # Direction of bisectors (perpendicular to p-p*)
    dx1, dy1 = -(y1s - y1), (x1s - x1)
    dx2, dy2 = -(y2s - y2), (x2s - x2)

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-12:
        return np.linspace(x1, x2, n_pts), np.linspace(y1, y2, n_pts)

    t = ((mx2 - mx1) * dy2 - (my2 - my1) * dx2) / denom
    cx = mx1 + t * dx1
    cy = my1 + t * dy1
    R = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)

    # Parametrise the arc from p1 to p2
    a1 = np.arctan2(y1 - cy, x1 - cx)
    a2 = np.arctan2(y2 - cy, x2 - cx)

    # Choose the shorter arc
    da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    angles = a1 + np.linspace(0, da, n_pts)
    xs = cx + R * np.cos(angles)
    ys = cy + R * np.sin(angles)
    return xs, ys


def _draw_poincare_tessellation(ax, depth=3):
    """Draw a {4,5}-like tessellation of the Poincaré disk using
    hyperbolic reflections, giving an Escher-like pattern that reveals
    how distances compress near the boundary."""
    from matplotlib.patches import Arc

    # Draw radial geodesics (lines through origin = diameters)
    n_radial = 12
    for i in range(n_radial):
        theta = np.pi * i / n_radial
        ax.plot([-np.cos(theta), np.cos(theta)],
                [-np.sin(theta), np.sin(theta)],
                color='#888888', linewidth=0.5, alpha=0.7, zorder=0)

    # Draw concentric horocircles (not true geodesics but illustrative)
    for r in [0.3, 0.55, 0.75, 0.88, 0.95]:
        c = plt.Circle((0, 0), r, fill=False, color='#888888',
                        linewidth=0.5, alpha=0.7, zorder=0)
        ax.add_patch(c)

    # Draw geodesic arcs orthogonal to the boundary for a richer tessellation
    # These are arcs of circles orthogonal to the unit disk
    n_arcs = 12
    for i in range(n_arcs):
        theta = 2 * np.pi * i / n_arcs
        # Points on the boundary that the geodesic connects
        for offset in [np.pi/3, np.pi/2, 2*np.pi/3]:
            t1 = theta
            t2 = theta + offset
            # Endpoints near boundary
            r_end = 0.98
            p1 = (r_end * np.cos(t1), r_end * np.sin(t1))
            p2 = (r_end * np.cos(t2), r_end * np.sin(t2))
            xs, ys = _poincare_geodesic_arc(p1, p2)
            # Clip to unit disk
            mask = xs**2 + ys**2 <= 1.0
            if np.any(mask):
                ax.plot(xs[mask], ys[mask], color='#888888',
                        linewidth=0.5, alpha=0.65, zorder=0)


def plot_hyperbolic_embedding():
    """Embedding hiperbólico de jerarquías (disco de Poincaré)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel izquierdo: Árbol en espacio euclidiano
    ax = axes[0]

    # Crear árbol jerárquico
    G = nx.balanced_tree(2, 3)  # árbol binario de profundidad 3

    # Layout jerárquico
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # Normalizar a círculo
    pos_array = np.array(list(pos.values()))
    pos_array = pos_array / np.max(np.abs(pos_array)) * 0.9
    pos = {i: pos_array[i] for i in range(len(pos))}

    # Colorear por profundidad
    depths = nx.single_source_shortest_path_length(G, 0)
    colors = [depths.get(i, 0) for i in G.nodes()]

    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap='viridis',
                           node_size=200, ax=ax)
    ax.text(0.5, -0.05, 'Embedding euclidiano', transform=ax.transAxes,
            ha='center', fontsize=11, fontstyle='italic')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Panel derecho: Disco de Poincaré con teselación
    ax = axes[1]

    # Simular posiciones en disco de Poincaré
    # Mapear profundidad a radio (más profundo = más cerca del borde)
    def depth_to_radius(d, max_d=3):
        # En hiperbólico, los niveles se acumulan cerca del borde
        return np.tanh(d * 0.8)

    # Calcular posiciones hiperbólicas
    pos_hyp = {}
    for node in G.nodes():
        d = depths.get(node, 0)
        r = depth_to_radius(d)
        # Ángulo basado en el índice del nodo en su nivel
        level_nodes = [n for n in G.nodes() if depths.get(n, 0) == d]
        idx = level_nodes.index(node)
        theta = 2 * np.pi * idx / len(level_nodes) + d * 0.3
        pos_hyp[node] = (r * np.cos(theta), r * np.sin(theta))

    # Disco exterior
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Teselación hiperbólica (geodésicas del disco de Poincaré)
    _draw_poincare_tessellation(ax)

    nx.draw_networkx_edges(G, pos_hyp, ax=ax, alpha=0.6, width=1.5)
    node_collection = nx.draw_networkx_nodes(G, pos_hyp, node_color=colors,
                                              cmap='viridis', node_size=200, ax=ax)
    node_collection.set_zorder(5)
    ax.text(0.5, -0.05, 'Embedding en disco de Poincaré',
            transform=ax.transAxes, ha='center', fontsize=11, fontstyle='italic')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperbolic_embedding.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ hyperbolic_embedding.png generado")


def plot_manifold_hypothesis():
    """Visualización de la hipótesis de la variedad"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    # Panel 1: Swiss roll (variedad 2D en 3D)
    ax1 = fig.add_subplot(121, projection='3d')

    n_samples = 1500
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    y = 21 * np.random.rand(n_samples)
    x = t * np.cos(t)
    z = t * np.sin(t)

    # Color por parámetro t
    ax1.scatter(x, y, z, c=t, cmap='viridis', s=10, alpha=0.6)
    ax1.set_title('Swiss Roll: Variedad 2D en R3')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Panel 2: Datos concentrados cerca de variedad
    ax2 = fig.add_subplot(122, projection='3d')

    # Superficie (variedad)
    u = np.linspace(-2, 2, 30)
    v = np.linspace(-2, 2, 30)
    U, V = np.meshgrid(u, v)
    X_surf = U
    Y_surf = V
    Z_surf = 0.3 * np.sin(U) * np.cos(V)

    ax2.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.3, cmap='Blues')

    # Puntos cerca de la superficie (datos)
    n_points = 500
    x_pts = np.random.uniform(-2, 2, n_points)
    y_pts = np.random.uniform(-2, 2, n_points)
    z_pts = 0.3 * np.sin(x_pts) * np.cos(y_pts) + 0.1 * np.random.randn(n_points)

    ax2.scatter(x_pts, y_pts, z_pts, c='red', s=10, alpha=0.6)
    ax2.set_title('Datos concentrados en variedad de baja dimension')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.suptitle('Hipótesis de la Variedad', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'manifold_hypothesis.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ manifold_hypothesis.png generado")


def plot_so2_irreps():
    """Representaciones irreducibles de SO(2)"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    theta = np.linspace(0, 2*np.pi, 200)

    n_values = [-2, -1, 0, 1, 2, 3]

    for ax, n in zip(axes.flatten(), n_values):
        # ρ_n(R_θ) = e^{inθ}
        rho_real = np.cos(n * theta)
        rho_imag = np.sin(n * theta)

        # Visualizar como curva paramétrica
        ax.plot(theta, rho_real, 'b-', label='Re(exp(in*theta))', linewidth=2)
        ax.plot(theta, rho_imag, 'r--', label='Im(exp(in*theta))', linewidth=2)

        ax.set_xlabel('theta')
        ax.set_ylabel('rho_n(R_theta)')
        ax.set_title(f'n = {n}: frecuencia {"nula" if n==0 else abs(n)}')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'pi/2', 'pi', '3pi/2', '2pi'])
        ax.grid(True, alpha=0.3)
        if n == -2:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Representaciones Irreducibles de SO(2): rho_n(R_theta) = exp(in*theta)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'so2_irreps.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ so2_irreps.png generado")


if __name__ == '__main__':
    print("Generando figuras para capítulo GDL...")
    print("=" * 50)

    plot_manifold_hypothesis()
    plot_graph_laplacian_eigenvectors()
    plot_eigenvalue_spectrum()
    plot_graph_fourier_transform()
    plot_hyperbolic_embedding()
    plot_so2_irreps()

    print("=" * 50)
    print("¡Todas las figuras generadas exitosamente!")
