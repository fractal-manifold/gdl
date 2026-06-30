#!/usr/bin/env python3
"""
Generate TikZ standalone figure for Graph Fourier Transform on C6.

Produces: graph_fourier_transform.tex

Usage:
    python generate_graph_fourier_tikz.py
"""

import numpy as np
from scipy.linalg import eigh
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_graph_fourier():
    """Compute Graph Fourier Transform on cyclic graph C6."""
    n = 6

    # Laplacian of C6
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 2
        L[i, (i + 1) % n] = -1
        L[i, (i - 1) % n] = -1

    eigenvalues, U = eigh(L)

    # Signal on graph
    signal = np.array([1.0, 0.5, 0.0, -0.5, -1.0, 0.0])

    # Fourier coefficients
    signal_hat = U.T @ signal

    # Partial reconstructions
    reconstructions = {}
    for k in [2, 4, 6]:
        reconstructions[k] = U[:, :k] @ signal_hat[:k]

    return eigenvalues, signal, signal_hat, reconstructions


def fmt_val(val, decimals=1):
    """Format a numeric value for display, avoiding -0.0."""
    rounded = round(val, decimals)
    if abs(rounded) < 10 ** (-decimals):
        rounded = 0.0
    if decimals == 1:
        return f"${rounded:+.1f}$" if abs(rounded) > 0.01 else "$0.0$"
    else:
        return f"${rounded:.2f}$" if abs(rounded) > 0.001 else "$0.00$"


def value_to_color(v, vmin=-1.0, vmax=1.0):
    """Map a scalar value to a TikZ color string (divergent blue-white-red)."""
    vnorm = np.clip(2.0 * (v - vmin) / (vmax - vmin) - 1.0, -1.0, 1.0)
    if abs(vnorm) < 0.05:
        return "white"
    elif vnorm >= 0:
        pct = int(round(vnorm * 80))
        return f"gdlred!{pct}!white"
    else:
        pct = int(round(abs(vnorm) * 80))
        return f"gdlblue!{pct}!white"


def value_to_border_color(v, vmin=-1.0, vmax=1.0):
    """Map a scalar value to a darker TikZ border color."""
    vnorm = np.clip(2.0 * (v - vmin) / (vmax - vmin) - 1.0, -1.0, 1.0)
    if abs(vnorm) < 0.05:
        return "gdlgray!60"
    elif vnorm >= 0:
        pct = int(round(max(vnorm * 80, 30)))
        return f"gdlred!{pct}!black!40"
    else:
        pct = int(round(max(abs(vnorm) * 80, 30)))
        return f"gdlblue!{pct}!black!40"


def generate_hexagon_nodes(signal_values, scope_shift, title, show_values=True,
                           vmin=-1.0, vmax=1.0):
    """Generate TikZ code for a hexagonal C6 graph with colored nodes."""
    lines = []
    lines.append(f"\\begin{{scope}}[shift={{{scope_shift}}}]")
    lines.append(f"  \\node[anchor=south, font=\\bfseries\\scriptsize] at (0, 1.7) {{{title}}};")

    # Node positions: regular hexagon, starting at top, going clockwise
    angles = [90, 30, -30, -90, -150, 150]
    radius = 1.1

    # Draw edges first (behind nodes)
    for i in range(6):
        j = (i + 1) % 6
        ax = radius * np.cos(np.radians(angles[i]))
        ay = radius * np.sin(np.radians(angles[i]))
        bx = radius * np.cos(np.radians(angles[j]))
        by = radius * np.sin(np.radians(angles[j]))
        lines.append(f"  \\draw[gdlgray!50, thick] ({ax:.4f},{ay:.4f}) -- ({bx:.4f},{by:.4f});")

    # Draw nodes
    for i in range(6):
        x = radius * np.cos(np.radians(angles[i]))
        y = radius * np.sin(np.radians(angles[i]))
        v = signal_values[i]
        fill = value_to_color(v, vmin, vmax)
        border = value_to_border_color(v, vmin, vmax)

        if show_values:
            label = fmt_val(v, 1)
        else:
            label = fmt_val(v, 2)

        lines.append(
            f"  \\node[circle, draw={border}, fill={fill}, "
            f"minimum size=14pt, inner sep=0pt, font=\\tiny] "
            f"(n{i}) at ({x:.4f},{y:.4f}) {{{label}}};"
        )

    # Node index labels (outside nodes)
    label_radius = radius + 0.4
    for i in range(6):
        lx = label_radius * np.cos(np.radians(angles[i]))
        ly = label_radius * np.sin(np.radians(angles[i]))
        lines.append(
            f"  \\node[font=\\tiny, text=gdlgray!80!black] at ({lx:.4f},{ly:.4f}) {{{i}}};"
        )

    lines.append("\\end{scope}")
    return "\n".join(lines)


def generate_bar_chart(values, scope_shift, title, fill_color_name, draw_color_name,
                       xlabel, ylabel, bar_width=0.6, ymin=None, ymax=None):
    """Generate TikZ/pgfplots code for a bar chart.

    fill_color_name and draw_color_name must be simple defined color names
    (e.g. 'barPurple', 'barPurpleDark') to avoid xcolor parsing issues.
    """
    n = len(values)
    if ymin is None:
        ymin = min(0, min(values) * 1.2)
    if ymax is None:
        ymax = max(0, max(values) * 1.2)
    if abs(ymax - ymin) < 0.01:
        ymax = 1.0

    lines = []
    lines.append(f"\\begin{{scope}}[shift={{{scope_shift}}}]")
    lines.append(f"  \\begin{{axis}}[")
    lines.append(f"    width=4.5cm, height=3.8cm,")
    lines.append(f"    ybar,")
    lines.append(f"    bar width={bar_width}em,")
    lines.append(f"    title={{\\scriptsize\\bfseries {title}}},")
    lines.append(f"    xlabel={{\\tiny {xlabel}}},")
    lines.append(f"    ylabel={{\\tiny {ylabel}}},")
    lines.append(f"    ymin={ymin:.2f}, ymax={ymax:.2f},")
    lines.append(f"    xtick={{{','.join(str(i) for i in range(n))}}},")
    lines.append(f"    xticklabel style={{font=\\tiny}},")
    lines.append(f"    yticklabel style={{font=\\tiny}},")
    lines.append(f"    xlabel style={{font=\\tiny}},")
    lines.append(f"    ylabel style={{font=\\tiny}},")
    lines.append(f"    title style={{font=\\scriptsize}},")
    lines.append(f"    tick label style={{font=\\tiny}},")
    lines.append(f"    axis lines=left,")
    lines.append(f"    enlarge x limits=0.15,")
    lines.append(f"  ]")

    coords = " ".join(f"({i},{values[i]:.6f})" for i in range(n))
    lines.append(f"    \\addplot+[fill={fill_color_name}, draw={draw_color_name}] "
                 f"coordinates {{{coords}}};")

    lines.append(f"  \\end{{axis}}")
    lines.append(f"\\end{{scope}}")
    return "\n".join(lines)


def generate_tex_file(eigenvalues, signal, signal_hat, reconstructions):
    """Generate the complete TikZ standalone .tex file."""

    parts = []

    # Preamble -- define simple color aliases to avoid xcolor parsing issues
    # with expressions like "gdlpurple!70" inside pgfplots styles
    parts.append(r"""% Transformada de Fourier en grafos sobre C_6
% Generado por: python generate_graph_fourier_tikz.py
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta, positioning, calc}

\definecolor{gdlblue}{RGB}{41, 98, 168}
\definecolor{gdlred}{RGB}{191, 49, 49}
\definecolor{gdlgreen}{RGB}{30, 130, 76}
\definecolor{gdlorange}{RGB}{217, 130, 30}
\definecolor{gdlpurple}{RGB}{118, 56, 138}
\definecolor{gdlgray}{RGB}{140, 140, 140}
\definecolor{gdlyellow}{RGB}{190, 140, 10}

% Pre-mixed colors for pgfplots bars (avoids xcolor expression issues)
\colorlet{barPurple}{gdlpurple!70}
\colorlet{barPurpleDark}{gdlpurple!90!black}
\colorlet{barGreen}{gdlgreen!70}
\colorlet{barGreenDark}{gdlgreen!90!black}

\begin{document}
\begin{tikzpicture}""")

    # Layout constants
    col_spacing = 5.5
    row_spacing = -5.0

    # ========== TOP ROW ==========

    # Panel 1: Signal on graph (top-left)
    parts.append("\n% === Row 1, Col 1: Signal on graph ===")
    parts.append(generate_hexagon_nodes(
        signal, "(0, 0)", r"Se\~nal $f$ en $C_6$",
        show_values=True, vmin=-1.0, vmax=1.0
    ))

    # Panel 2: Fourier coefficients (top-center)
    # Shift bar charts down by -1.6 so their visual center aligns with graphs
    parts.append("\n% === Row 1, Col 2: Fourier coefficients ===")
    fhat_min = min(signal_hat) * 1.3
    fhat_max = max(signal_hat) * 1.3
    parts.append(generate_bar_chart(
        signal_hat, f"({col_spacing - 0.3}, -1.6)",
        r"Coeficientes $\hat{f}_k = U^\top f$",
        "barPurple", "barPurpleDark",
        "$k$", r"$\hat{f}_k$",
        ymin=fhat_min, ymax=fhat_max
    ))

    # Panel 3: Eigenvalue spectrum (top-right)
    parts.append("\n% === Row 1, Col 3: Eigenvalue spectrum ===")
    parts.append(generate_bar_chart(
        eigenvalues, f"({2 * col_spacing - 0.3}, -1.6)",
        r"Espectro $\lambda_k$",
        "barGreen", "barGreenDark",
        "$k$", r"$\lambda_k$",
        ymin=0, ymax=max(eigenvalues) * 1.15
    ))

    # ========== BOTTOM ROW ==========
    parts.append("\n% === Row 2: Partial reconstructions ===")

    for idx, k in enumerate([2, 4, 6]):
        rec = reconstructions[k]
        shift_x = idx * col_spacing
        shift_y = row_spacing

        if k < 6:
            title = f"Reconstrucci\\'on $k={k}$ modos"
        else:
            title = f"Reconstrucci\\'on completa ($k={k}$)"

        parts.append(generate_hexagon_nodes(
            rec, f"({shift_x}, {shift_y})",
            title,
            show_values=False, vmin=-1.0, vmax=1.0
        ))

    # ========== COLOR LEGEND ==========
    parts.append("\n% === Color legend ===")
    legend_y = row_spacing - 2.2
    parts.append(f"\\begin{{scope}}[shift={{(0, {legend_y})}}]")
    # Draw a horizontal color bar
    n_steps = 20
    bar_width = 6.0
    bar_height = 0.25
    start_x = col_spacing - bar_width / 2
    step = bar_width / n_steps
    for i in range(n_steps):
        v = -1.0 + 2.0 * (i + 0.5) / n_steps
        col = value_to_color(v)
        x = start_x + i * step
        parts.append(
            f"  \\fill[{col}] ({x:.3f}, 0) rectangle ({x + step:.3f}, {bar_height});")
    # Border
    parts.append(
        f"  \\draw[gdlgray!60] ({start_x:.3f}, 0) rectangle "
        f"({start_x + bar_width:.3f}, {bar_height});")
    # Labels
    parts.append(
        f"  \\node[font=\\tiny, anchor=north] at ({start_x:.3f}, -0.05) {{$-1$}};")
    parts.append(
        f"  \\node[font=\\tiny, anchor=north] at ({start_x + bar_width / 2:.3f}, -0.05) {{$0$}};")
    parts.append(
        f"  \\node[font=\\tiny, anchor=north] at ({start_x + bar_width:.3f}, -0.05) {{$+1$}};")
    parts.append(
        f"  \\node[font=\\tiny, anchor=north] at ({start_x + bar_width / 2:.3f}, -0.4) "
        f"{{Valor de la se\\~nal}};")
    parts.append("\\end{scope}")

    # Close
    parts.append(r"""
\end{tikzpicture}
\end{document}""")

    return "\n".join(parts)


def main():
    eigenvalues, signal, signal_hat, reconstructions = compute_graph_fourier()

    print("=== Graph Fourier Transform on C6 ===")
    print(f"Signal:       {signal}")
    print(f"Eigenvalues:  {np.round(eigenvalues, 4)}")
    print(f"Fourier coef: {np.round(signal_hat, 4)}")
    for k in [2, 4, 6]:
        print(f"Recon k={k}:   {np.round(reconstructions[k], 4)}")

    tex_content = generate_tex_file(eigenvalues, signal, signal_hat, reconstructions)

    output_path = os.path.join(OUTPUT_DIR, "graph_fourier_transform.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"\nGenerated: {output_path}")


if __name__ == "__main__":
    main()
