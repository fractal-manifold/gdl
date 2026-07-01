---
title: Artificial Neural Networks
order: 2
part: Foundations
summary: The perceptron, layers as measurable maps, and a full proof of Cybenko's universal approximation theorem — with the parametric limits of Euclidean architectures.
tags: ['Perceptron', 'Universal approximation', 'Deep learning']
draft: false
---

## The artificial neuron

The chapter formalises the classical building blocks of deep learning before
the geometric picture is introduced. A neuron composes an affine map with a
non-linearity,

$$
y = \sigma\!\left(\langle w, x\rangle + b\right),
$$

and layers stack these into vector-valued measurable functions between feature
spaces. This measure-theoretic footing is what later makes precise statements
about approximation and equivariance possible.

## Data and its structure

Data is described by the domain it lives on — grids, sequences, tables — and by
the symmetries that act on that domain. Making this structure explicit is the
bridge to the geometric framework: it is exactly the structure that Euclidean
architectures leave on the table.

## Universal approximation

The theoretical centrepiece is a complete proof of **Cybenko's universal
approximation theorem**: a single hidden layer with a sigmoidal activation can
approximate any continuous function on a compact set to arbitrary accuracy.
This guarantees expressive power in principle, and becomes the baseline that
the equivariant universality results of later chapters generalise.

## Networks as graphs

Casting a network as a computation graph — neurons as nodes, weights as edges —
previews the unifying theme of the book: a plain neural network is message
passing over a **complete graph**, where every unit talks to every other.

## Learning and deep networks

Backpropagation, gradient-based optimisation and the ingredients of *deep*
learning (depth, data, hardware) are developed here. The chapter closes on the
inherent limitation of these Euclidean architectures: they cannot exploit the
symmetries of structured data, and so pay for it in parameters and data
efficiency — the motivation for everything that follows.
