---
title: Convolutional Models
order: 4
part: Models
summary: From classical CNNs to group, steerable, spherical, E(3)/SE(3), gauge and Clifford convolutions — culminating in an equivariant universality theorem.
tags: ['Group convolution', 'Steerable CNNs', 'Gauge equivariance']
draft: false
---

## Convolution as the equivariant map

The first family of geometric architectures generalises the convolution. On a
regular grid, convolution is precisely the linear map that is **equivariant to
translations** — the reason CNNs are so effective on images. The chapter builds
a ladder of generalisations that replace translation with richer symmetry
groups.

## Harmonic analysis and spectral convolutions

Harmonic analysis is the unifying tool: the Fourier transform, spherical
harmonics and Chebyshev decompositions connect the spatial and spectral views
of convolution, and make group convolution computable.

## A ladder of generalisations

- **Group convolutions (G-CNNs)** for finite groups — equivariance to rotations and reflections.
- **Steerable CNNs** for continuous $\mathrm{SO}(2)$, built from irreducible representations.
- **Spherical CNNs** for $\mathrm{SO}(3)$, for signals on the sphere.
- **$\mathrm{E}(3)$/$\mathrm{SE}(3)$-equivariant networks** for 3D point clouds and molecules.
- **Gauge-equivariant convolutions** on manifolds, using local frames and parallel transport.
- **Clifford / geometric-algebra convolutions** acting directly on multivectors.

## Equivariant universality

The ladder culminates in an **equivariant universality theorem**: equivariant
CNNs can approximate any continuous equivariant function — the geometric
counterpart of the classical universal approximation result. Topological
extensions via **Hodge Laplacians** push convolution onto simplicial and
cellular complexes, capturing higher-order relations beyond pairwise edges.
