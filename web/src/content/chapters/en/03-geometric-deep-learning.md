---
title: Geometric Deep Learning
order: 3
part: The Framework
summary: The unifying framework — the 5 Gs, equivariance and invariance, and equivariant message passing that recovers CNNs, GNNs and Transformers as special cases.
tags: ['Equivariance', 'Message passing', 'Weisfeiler–Leman']
draft: false
---

## The paradigm

This is the theoretical heart of the book: a single framework from which the
dominant architectures fall out as special cases. Its organising principle is
**symmetry** — the choice of a geometric domain, a symmetry group acting on it,
and maps that commute with that action.

## The five Gs, in depth

- **Geometry — the domains of data.** Differentiable and Riemannian manifolds, tangent spaces and curvature give a precise language for the space a signal lives on.
- **Groups — the symmetries of data.** Lie groups and their algebras describe symmetries; representation theory — the **Peter–Weyl theorem** and **Schur's lemma** — decomposes them into irreducible pieces, the raw material for equivariant operators.
- **Graphs — the discretisation of geometries.** The graph Laplacian and spectral analysis turn continuous domains into computable, neighbourhood-preserving discretisations.
- **Geodesics — information propagation.** Metrics, the exponential map and parallel transport formalise how information travels — the geometric root of message passing.
- **Gauges — local invariances.** Principal bundles and connections handle the freedom to choose local coordinate frames.

## Equivariance and invariance

Equivariance and invariance are formalised as compatibility conditions between
a function and a group action. Equivariance,

$$
f(g \cdot x) = \rho(g)\, f(x),
$$

is the guiding design principle: it is what lets a network respect the
structure of its data by construction rather than by training.

## The unifying synthesis

The chapter's synthesis is **equivariant message passing** as a common
denominator:

- plain neural networks are message passing over a complete graph;
- **CNNs** are message passing over a regular grid, with weights tied by the translation group;
- **Transformers** are *adaptive* message passing, where communication weights are computed from the data.

The **Weisfeiler–Leman** test pins down the expressive limit of this family:
the graph-distinguishing power of message-passing networks coincides with the
1-WL algorithm — a precise ceiling, and a map of how the architectural families
include one another.
