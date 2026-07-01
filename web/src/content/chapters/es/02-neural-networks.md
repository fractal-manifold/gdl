---
title: Redes neuronales artificiales
order: 2
part: Fundamentos
summary: El perceptrón, las capas como aplicaciones medibles y una demostración completa del teorema de aproximación universal de Cybenko, con los límites paramétricos de las arquitecturas euclídeas.
tags: ['Perceptrón', 'Aproximación universal', 'Aprendizaje profundo']
draft: false
---

## La neurona artificial

El capítulo formaliza los bloques clásicos del aprendizaje profundo antes de
introducir la perspectiva geométrica. Una neurona compone una aplicación afín
con una no linealidad,

$$
y = \sigma\!\left(\langle w, x\rangle + b\right),
$$

y las capas las apilan en funciones medibles con valores vectoriales entre
espacios de características. Esta base de teoría de la medida es lo que permite
después enunciar con precisión resultados sobre aproximación y equivariancia.

## Los datos y su estructura

Los datos se describen por el dominio en que viven —rejillas, secuencias,
tablas— y por las simetrías que actúan sobre ese dominio. Hacer explícita esa
estructura es el puente hacia el marco geométrico: es exactamente la estructura
que las arquitecturas euclídeas desaprovechan.

## Aproximación universal

La pieza teórica central es una demostración completa del **teorema de
aproximación universal de Cybenko**: una única capa oculta con activación
sigmoidal puede aproximar cualquier función continua en un compacto con
precisión arbitraria. Esto garantiza la capacidad expresiva en principio y se
convierte en la base que los resultados de universalidad equivariante de
capítulos posteriores generalizan.

## Las redes como grafos

Ver una red como un grafo de cómputo —neuronas como nodos, pesos como aristas—
anticipa el hilo unificador del libro: una red neuronal simple es paso de
mensajes sobre un **grafo completo**, donde cada unidad habla con todas las
demás.

## Aprendizaje y redes profundas

Aquí se desarrollan la retropropagación, la optimización por gradiente y los
ingredientes del aprendizaje *profundo* (profundidad, datos, hardware). El
capítulo cierra con la limitación inherente de estas arquitecturas euclídeas: no
pueden explotar las simetrías de los datos estructurados y lo pagan en
parámetros y eficiencia de datos, la motivación de todo lo que sigue.
