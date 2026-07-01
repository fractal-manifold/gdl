---
title: Conclusiones
order: 6
part: Cierre
summary: La geometría como principio organizador — cómo dominio, grupo de simetría y conectividad separan ANN, CNN, GNN y Transformers, y dónde están los problemas abiertos.
tags: ['Síntesis', 'Problemas abiertos', 'Direcciones futuras']
draft: false
---

## Qué nos da el marco

El arco del libro va de las redes neuronales tradicionales, pasando por el marco
geométrico de las 5 Ges, hasta la convolución y la atención como sus dos grandes
familias. La afirmación unificadora es concreta: las diferencias entre **ANN,
CNN, GNN y Transformers** se reducen a tres elecciones —el **dominio
geométrico**, el **grupo de simetría** y la **estructura de conectividad**. El
paso de mensajes equivariante es la implementación común, y la universalidad
equivariante garantiza su capacidad expresiva.

## Problemas abiertos

- **La barrera de Weisfeiler–Leman** — las redes de paso de mensajes estándar heredan el techo expresivo del test 1-WL; superarlo a coste polinómico sigue abierto.
- **Rigidez frente a adaptabilidad** — dónde situarse en el espectro CNN–atención para un dominio dado no tiene aún respuesta general.
- **Equivariancia gauge y topología global** — la simetría gauge local es elegante, pero las obstrucciones topológicas globales dificultan las implementaciones a escala.
- **Estructuras de orden superior** — las arquitecturas sobre complejos simpliciales y celulares aún son inmaduras frente a las redes de grafos.

## Direcciones futuras

El libro apunta hacia arquitecturas híbridas CNN–atención que interpolen entre
rigidez y adaptabilidad, modelos sobre geometrías de curvatura mixta y espacios
producto, redes neuronales de haces (*sheaf neural networks*) y análisis armónico
generalizado más allá de los grupos compactos. La reflexión final es que la
geometría no es solo un truco de rendimiento, sino un **principio organizador**
—una teoría con capacidad predictiva para el *diseño fundamentado de futuras
arquitecturas*, no un mero catálogo de las existentes.
