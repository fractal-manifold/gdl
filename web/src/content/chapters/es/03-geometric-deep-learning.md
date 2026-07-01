---
title: Aprendizaje profundo geométrico
order: 3
part: El marco
summary: El marco unificador — las 5 Ges, equivariancia e invariancia, y el paso de mensajes equivariante que recupera CNN, GNN y Transformers como casos particulares.
tags: ['Equivariancia', 'Paso de mensajes', 'Weisfeiler–Leman']
draft: false
---

## El paradigma

Este es el corazón teórico del libro: un único marco del que se derivan las
arquitecturas dominantes como casos particulares. Su principio organizador es la
**simetría** —la elección de un dominio geométrico, un grupo de simetría que
actúa sobre él y aplicaciones que conmutan con esa acción.

## Las cinco Ges, en profundidad

- **Geometry — los dominios de los datos.** Variedades diferenciables y de Riemann, espacios tangentes y curvatura dan un lenguaje preciso para el espacio en que vive una señal.
- **Groups — las simetrías de los datos.** Los grupos de Lie y sus álgebras describen simetrías; la teoría de representaciones —el **teorema de Peter–Weyl** y el **lema de Schur**— las descompone en piezas irreducibles, la materia prima de los operadores equivariantes.
- **Graphs — la discretización de geometrías.** El laplaciano del grafo y el análisis espectral convierten dominios continuos en discretizaciones computables que preservan la vecindad.
- **Geodesics — la propagación de la información.** Métricas, mapa exponencial y transporte paralelo formalizan cómo viaja la información: la raíz geométrica del paso de mensajes.
- **Gauges — invarianzas locales.** Fibrados principales y conexiones manejan la libertad de elegir marcos de coordenadas locales.

## Equivariancia e invariancia

La equivariancia y la invariancia se formalizan como condiciones de
compatibilidad entre una función y una acción de grupo. La equivariancia,

$$
f(g \cdot x) = \rho(g)\, f(x),
$$

es el principio de diseño rector: es lo que permite a una red respetar la
estructura de sus datos por construcción y no por entrenamiento.

## La síntesis unificadora

La síntesis del capítulo es el **paso de mensajes equivariante** como común
denominador:

- las redes neuronales simples son paso de mensajes sobre un grafo completo;
- las **CNN** son paso de mensajes sobre una rejilla regular, con pesos atados por el grupo de traslación;
- los **Transformers** son paso de mensajes *adaptativo*, donde los pesos de comunicación se calculan a partir de los datos.

El test de **Weisfeiler–Leman** fija el límite expresivo de esta familia: el
poder de las redes de paso de mensajes para distinguir grafos coincide con el
algoritmo 1-WL —un techo preciso y un mapa de cómo las familias arquitectónicas
se contienen unas a otras.
