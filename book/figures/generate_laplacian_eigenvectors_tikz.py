#!/usr/bin/env python3
"""Generate TikZ code for Laplacian eigenvector visualization on 8x8 grid."""
import numpy as np
from scipy.linalg import eigh
import os


def main():
    n = 8
    N = n * n  # 64 nodes

    # Build Laplacian of 8x8 grid graph
    L = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            degree = 0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni_, nj_ = i + di, j + dj
                if 0 <= ni_ < n and 0 <= nj_ < n:
                    nidx = ni_ * n + nj_
                    L[idx, nidx] = -1
                    degree += 1
            L[idx, idx] = degree

    eigenvalues, eigenvectors = eigh(L)

    # Select eigenvector indices: low freq [1,2,3], high freq [-3,-2,-1]
    indices = [1, 2, 3, N - 3, N - 2, N - 1]

    # Layout parameters
    node_spacing = 0.45  # cm between adjacent nodes
    panel_hspace = 4.5   # horizontal center-to-center between panels
    panel_vspace = 5.0   # vertical center-to-center between rows
    node_radius = 2.5    # pt

    # Panel positions (col, row) -> (x_offset, y_offset)
    panel_positions = []
    for panel_idx in range(6):
        col = panel_idx % 3
        row = panel_idx // 3
        x_off = col * panel_hspace
        y_off = -row * panel_vspace
        panel_positions.append((x_off, y_off))

    # Start building TikZ code
    lines = []
    lines.append(r"\documentclass[tikz,border=10pt]{standalone}")
    lines.append(r"\usepackage{tikz}")
    lines.append(r"\usepackage{amsmath}")
    lines.append("")
    lines.append(r"\definecolor{gdlblue}{RGB}{41, 98, 168}")
    lines.append(r"\definecolor{gdlred}{RGB}{191, 49, 49}")
    lines.append(r"\definecolor{gdlgreen}{RGB}{30, 130, 76}")
    lines.append(r"\definecolor{gdlorange}{RGB}{217, 130, 30}")
    lines.append(r"\definecolor{gdlpurple}{RGB}{118, 56, 138}")
    lines.append(r"\definecolor{gdlgray}{RGB}{140, 140, 140}")
    lines.append(r"\definecolor{gdlyellow}{RGB}{190, 140, 10}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{tikzpicture}")
    lines.append("")

    for panel_idx, eig_idx in enumerate(indices):
        x_off, y_off = panel_positions[panel_idx]
        row = panel_idx // 3  # 0 = low freq, 1 = high freq
        eigval = eigenvalues[eig_idx]
        eigvec = eigenvectors[:, eig_idx]

        # Normalize eigenvector to [-1, 1]
        max_abs = np.max(np.abs(eigvec))
        if max_abs > 1e-12:
            eigvec_norm = eigvec / max_abs
        else:
            eigvec_norm = eigvec

        lines.append(f"  % === Panel {panel_idx + 1}: eigenvector {eig_idx}, "
                      f"lambda = {eigval:.4f} ===")
        lines.append(f"  \\begin{{scope}}[shift={{({x_off:.2f},{y_off:.2f})}}]")

        # Draw edges first (behind nodes)
        lines.append("    % Edges")
        for i in range(n):
            for j in range(n):
                x = j * node_spacing
                y = (n - 1 - i) * node_spacing  # flip y so row 0 is at top
                # Right neighbor
                if j + 1 < n:
                    x2 = (j + 1) * node_spacing
                    y2 = y
                    lines.append(
                        f"    \\draw[gdlgray!30, very thin] "
                        f"({x:.3f},{y:.3f}) -- ({x2:.3f},{y2:.3f});"
                    )
                # Down neighbor (which is up in flipped coords)
                if i + 1 < n:
                    x2 = x
                    y2 = (n - 1 - (i + 1)) * node_spacing
                    lines.append(
                        f"    \\draw[gdlgray!30, very thin] "
                        f"({x:.3f},{y:.3f}) -- ({x2:.3f},{y2:.3f});"
                    )

        # Draw nodes
        lines.append("    % Nodes")
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                v = eigvec_norm[idx]
                x = j * node_spacing
                y = (n - 1 - i) * node_spacing

                intensity = int(abs(v) * 80)
                border_intensity = min(intensity + 20, 100)

                if v >= 0:
                    fill_color = f"gdlred!{intensity}!white"
                    draw_color = f"gdlred!{border_intensity}"
                else:
                    fill_color = f"gdlblue!{intensity}!white"
                    draw_color = f"gdlblue!{border_intensity}"

                lines.append(
                    f"    \\fill[{fill_color}, draw={draw_color}, "
                    f"line width=0.2pt] "
                    f"({x:.3f},{y:.3f}) circle ({node_radius}pt);"
                )

        # Title: eigenvalue
        grid_width = (n - 1) * node_spacing
        grid_height = (n - 1) * node_spacing
        title_x = grid_width / 2
        title_y = grid_height + 0.45

        lines.append("    % Title")
        lines.append(
            f"    \\node[anchor=south, font=\\scriptsize] at "
            f"({title_x:.3f},{title_y:.3f}) "
            f"{{$\\lambda_{{{eig_idx}}} = {eigval:.2f}$}};"
        )

        # Frequency label below
        label_y = -0.5
        if row == 0:
            label_color = "gdlblue!70!black"
            label_text = "Baja frecuencia"
        else:
            label_color = "gdlred!70!black"
            label_text = "Alta frecuencia"

        lines.append("    % Frequency label")
        lines.append(
            f"    \\node[anchor=north, font=\\scriptsize, "
            f"text={label_color}] at "
            f"({title_x:.3f},{label_y:.3f}) {{{label_text}}};"
        )

        lines.append(r"  \end{scope}")
        lines.append("")

    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")

    tex_code = "\n".join(lines) + "\n"

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "laplacian_eigenvectors.tex",
    )
    with open(output_path, "w") as f:
        f.write(tex_code)

    print(f"Written {output_path}")
    print(f"  Total lines: {len(lines)}")
    print(f"  Eigenvalues used:")
    for eig_idx in indices:
        print(f"    lambda_{eig_idx} = {eigenvalues[eig_idx]:.4f}")


if __name__ == "__main__":
    main()
