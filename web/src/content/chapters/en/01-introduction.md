---
title: Introduction
order: 1
part: Foundations
summary: Why Euclidean deep learning hits a wall on graphs, manifolds and groups — and how geometric priors invert the data–architecture relationship.
tags: ['Motivation', 'The 5 Gs', 'Equivariance']
draft: false
---

## From Euclidean success to its limits

Deep learning transformed artificial intelligence, but the architectures
behind that success — convolutional, recurrent and attention networks — were
conceived for data on **regular Euclidean domains**: pixel grids, temporal
sequences, feature tables. As the field turned to molecules, social networks
and 3D surfaces, the assumption of Euclidean regularity became a *conceptual*
limitation, not merely a technical one. Forcing a graph into an adjacency
matrix or a surface into voxels discards structure and wastes parameters.

## Geometric Deep Learning

Geometric Deep Learning (GDL) inverts the traditional relationship between data
and architecture: instead of forcing the data to fit a preexisting model, the
**geometric structure of the data dictates the structure of the model**. By
building symmetries directly into the network, GDL earns equivariance, robust
generalisation and parameter efficiency.

## The five Gs

The book is organised around five interlocking concepts — the *5 Gs*:

- **Geometry** — the intrinsic structure of the data domain (Riemannian manifolds, tangent spaces, curvature).
- **Groups** — the symmetries that preserve the problem (Lie groups, representation theory, the equivariance principle).
- **Graphs** — discretisations of geometric domains that capture neighbourhood relations (the graph Laplacian, spectral analysis).
- **Geodesics** — optimal paths that drive information propagation (metrics, the exponential map, parallel transport — message passing in GNNs).
- **Gauges** — local invariances in the choice of coordinate frames (principal bundles and connections).

A function $f$ is *equivariant* to a group $\mathfrak{G}$ acting on its input
and output spaces when

$$
f(g \cdot x) = g \cdot f(x) \qquad \forall\, g \in \mathfrak{G}.
$$

## What this book sets out to do

- Develop a **rigorous mathematical framework** for the 5 Gs, grounded in group theory, differential geometry and harmonic analysis.
- **Unify architectures**: show formally how CNNs, GNNs and Transformers emerge as special cases of equivariant message passing.
- **Extend** convolution and attention to non-Euclidean domains.
- **Connect theory to practice** through representative modern architectures.

## Roadmap

The remaining chapters build from foundations to advanced models: artificial
neural networks and universal approximation, the unifying GDL framework, then
its two great families — convolutional and attention models — before a closing
synthesis of results, limitations and open problems.
