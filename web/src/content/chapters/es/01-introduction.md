---
title: Introducción
order: 1
part: Fundamentos
summary: Por qué el aprendizaje profundo euclídeo choca con grafos, variedades y grupos, y cómo los priores geométricos invierten la relación datos–arquitectura.
tags: ['Motivación', 'Las 5 Ges', 'Equivariancia']
draft: false
---

## Del éxito euclídeo a sus límites

El aprendizaje profundo transformó la inteligencia artificial, pero las
arquitecturas responsables de ese éxito —convolucionales, recurrentes y de
atención— se concibieron para datos en **dominios euclídeos regulares**:
rejillas de píxeles, secuencias temporales, tablas de características. Al aplicar
estas técnicas a moléculas, redes sociales o superficies 3D, la suposición de
regularidad euclídea se reveló como una limitación *conceptual*, no solo
técnica. Forzar un grafo a una matriz de adyacencia o una superficie a vóxeles
descarta estructura y desperdicia parámetros.

## Aprendizaje profundo geométrico

El aprendizaje profundo geométrico (GDL) invierte la relación tradicional entre
datos y arquitectura: en lugar de forzar los datos a un modelo preexistente, es
la **estructura geométrica de los datos la que dicta la estructura del modelo**.
Al incorporar las simetrías directamente en la red, el GDL obtiene
equivariancia, generalización robusta y eficiencia en parámetros.

## Las cinco Ges

El libro se organiza en torno a cinco conceptos entrelazados —las *5 Ges*:

- **Geometry** (geometría) — la estructura intrínseca del dominio (variedades de Riemann, espacios tangentes, curvatura).
- **Groups** (grupos) — las simetrías que preservan el problema (grupos de Lie, teoría de representaciones, principio de equivariancia).
- **Graphs** (grafos) — discretizaciones de dominios geométricos que capturan relaciones de vecindad (laplaciano del grafo, análisis espectral).
- **Geodesics** (geodésicas) — caminos óptimos que gobiernan la propagación de la información (métricas, mapa exponencial, transporte paralelo — el paso de mensajes en las GNN).
- **Gauges** (gauges) — invarianzas locales en la elección de marcos de coordenadas (fibrados principales y conexiones).

Una función $f$ es *equivariante* a un grupo $\mathfrak{G}$ que actúa sobre sus
espacios de entrada y salida cuando

$$
f(g \cdot x) = g \cdot f(x) \qquad \forall\, g \in \mathfrak{G}.
$$

## Qué persigue este libro

- Desarrollar un **marco matemático riguroso** para las 5 Ges, sobre teoría de grupos, geometría diferencial y análisis armónico.
- **Unificar arquitecturas**: mostrar formalmente cómo CNN, GNN y Transformers surgen como casos particulares del paso de mensajes equivariante.
- **Extender** la convolución y la atención a dominios no euclídeos.
- **Conectar teoría y práctica** a través de arquitecturas modernas representativas.

## Hoja de ruta

Los capítulos siguientes van de los fundamentos a los modelos avanzados: redes
neuronales artificiales y aproximación universal, el marco unificador del GDL y
sus dos grandes familias —modelos convolucionales y de atención— antes de una
síntesis final de resultados, limitaciones y problemas abiertos.
