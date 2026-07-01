---
title: Conclusions
order: 6
part: Closing
summary: Geometry as an organizing principle — how domain, symmetry group and connectivity separate ANNs, CNNs, GNNs and Transformers, and where the open problems lie.
tags: ['Synthesis', 'Open problems', 'Future directions']
draft: false
---

## What the framework buys us

The arc of the book runs from traditional neural networks, through the
geometric framework of the 5 Gs, to convolution and attention as its two great
families. The unifying claim is concrete: the differences between **ANNs,
CNNs, GNNs and Transformers** reduce to three choices — the **geometric
domain**, the **symmetry group**, and the **connectivity structure**.
Equivariant message passing is the common implementation, and equivariant
universality guarantees its expressive power.

## Open problems

- **The Weisfeiler–Leman barrier** — standard message-passing networks inherit the expressive ceiling of the 1-WL test; surpassing it at polynomial cost is open.
- **Rigidity versus adaptability** — where to sit on the CNN–attention spectrum for a given domain has no general answer yet.
- **Gauge equivariance and global topology** — local gauge symmetry is elegant, but global topological obstructions make scalable implementations hard.
- **Higher-order structures** — architectures over simplicial and cellular complexes are still immature compared to graph networks.

## Future directions

The book points toward hybrid CNN–attention architectures that interpolate
between rigidity and adaptability, models over mixed-curvature and product
geometries, sheaf neural networks, and generalised harmonic analysis beyond
compact groups. The closing reflection is that geometry is not just a
performance trick but an **organising principle** — a theory with predictive
power for the *principled design of future architectures*, not merely a
taxonomy of existing ones.
