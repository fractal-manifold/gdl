---
title: Modelos convolucionales
order: 4
part: Modelos
summary: De las CNN clásicas a las convoluciones de grupo, orientables, esféricas, E(3)/SE(3), gauge y de Clifford — culminando en un teorema de universalidad equivariante.
tags: ['Convolución de grupo', 'CNN orientables', 'Equivariancia gauge']
draft: false
---

## La convolución como aplicación equivariante

La primera familia de arquitecturas geométricas generaliza la convolución. En
una rejilla regular, la convolución es precisamente la aplicación lineal
**equivariante a las traslaciones** —la razón de la eficacia de las CNN en
imágenes. El capítulo construye una escalera de generalizaciones que sustituyen
la traslación por grupos de simetría más ricos.

## Análisis armónico y convoluciones espectrales

El análisis armónico es la herramienta unificadora: la transformada de Fourier,
los armónicos esféricos y las descomposiciones de Chebyshev conectan las visiones
espacial y espectral de la convolución y hacen computable la convolución de
grupo.

## Una escalera de generalizaciones

- **Convoluciones de grupo (G-CNN)** para grupos finitos — equivariancia a rotaciones y reflexiones.
- **CNN orientables (steerable)** para $\mathrm{SO}(2)$ continuo, construidas a partir de representaciones irreducibles.
- **CNN esféricas** para $\mathrm{SO}(3)$, para señales sobre la esfera.
- **Redes equivariantes $\mathrm{E}(3)$/$\mathrm{SE}(3)$** para nubes de puntos y moléculas en 3D.
- **Convoluciones equivariantes gauge** sobre variedades, con marcos locales y transporte paralelo.
- **Convoluciones de Clifford / álgebra geométrica** que actúan directamente sobre multivectores.

## Universalidad equivariante

La escalera culmina en un **teorema de universalidad equivariante**: las CNN
equivariantes pueden aproximar cualquier función continua equivariante —el
análogo geométrico del resultado clásico de aproximación universal. Las
extensiones topológicas vía **laplacianos de Hodge** llevan la convolución a
complejos simpliciales y celulares, capturando relaciones de orden superior más
allá de las aristas por pares.
