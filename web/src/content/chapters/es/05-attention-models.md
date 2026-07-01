---
title: Modelos de atención
order: 5
part: Modelos
summary: Los Transformers como paso de mensajes adaptativo, y después atención equivariante, hiperbólica y topológica — GATr, laplacianos de Hodge y homología persistente.
tags: ['Transformers', 'Atención equivariante', 'Geometría hiperbólica']
draft: false
---

## La atención como paso de mensajes adaptativo

Donde la convolución ata sus pesos rígidamente a un grupo de simetría, la
**atención** calcula los pesos de comunicación de forma dinámica a partir de los
datos. La autoatención es, por tanto, paso de mensajes *adaptativo* sobre un
grafo completo —la misma lente unificadora del capítulo convolucional, desde el
extremo opuesto del espectro rigidez–adaptabilidad.

## La geometría se encuentra con la atención

El capítulo fundamenta los Transformers en el mismo instrumental geométrico y
después los extiende:

- **Teoría de grupos y equivariancia** — atención que respeta grupos de simetría.
- **Análisis espectral** — la visión de la atención mediante el laplaciano del grafo en dominios estructurados.
- **Geometría diferencial** — atención sobre variedades y **atención hiperbólica** para datos con estructura jerárquica, en espacios de curvatura negativa constante.
- **Álgebra geométrica** — atención equivariante sobre multivectores de Clifford, como en **GATr** (Geometric Algebra Transformers).
- **Topología algebraica** — el **laplaciano de Hodge** y la **homología persistente** aportan estructura topológica y de orden superior a la atención.

## Rigidez frente a adaptabilidad

Un tema recurrente es el **compromiso entre rigidez y adaptabilidad**: las CNN
equivariantes fijan la simetría a cambio de eficiencia de datos y garantías; la
atención las cambia por flexibilidad y expresividad a mayor coste de datos. Un
teorema de correspondencia —y una conjetura de convergencia— formalizan cómo se
relacionan los paradigmas convolucional y de atención, y cuándo se encuentran.
