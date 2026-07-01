---
title: Attention Models
order: 5
part: Models
summary: Transformers as adaptive message passing, then equivariant, hyperbolic and topological attention — GATr, Hodge Laplacians and persistent homology.
tags: ['Transformers', 'Equivariant attention', 'Hyperbolic geometry']
draft: false
---

## Attention as adaptive message passing

Where convolution ties its weights rigidly to a symmetry group, **attention**
computes communication weights dynamically from the data. Self-attention is
therefore *adaptive* message passing on a complete graph — the same unifying
lens as the convolutional chapter, from the opposite end of the
rigidity–adaptability spectrum.

## Geometry meets attention

The chapter grounds Transformers in the same geometric toolkit and then extends
them:

- **Group theory and equivariance** — attention that respects symmetry groups.
- **Spectral analysis** — the graph-Laplacian view of attention on structured domains.
- **Differential geometry** — attention on manifolds and **hyperbolic attention** for data with hierarchical structure, in spaces of constant negative curvature.
- **Geometric algebra** — equivariant attention over Clifford multivectors, as in **GATr** (Geometric Algebra Transformers).
- **Algebraic topology** — the **Hodge Laplacian** and **persistent homology** bring higher-order and topological structure into attention.

## Rigidity versus adaptability

A recurring theme is the **trade-off between rigidity and adaptability**:
equivariant CNNs bake in symmetry for data efficiency and guarantees; attention
trades those for flexibility and expressiveness at higher data cost. A
correspondence theorem — and a convergence conjecture — formalise how the
convolutional and attention paradigms relate, and when they meet.
