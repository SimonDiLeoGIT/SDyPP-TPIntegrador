# Sistemas distribuidos y programación paralela 2024

## [Trabajo práctico integrador](https://docs.google.com/document/d/14f0_gtVplWhJ0CAvfwddja1G_Ed--CjNQREekO_lZQs/edit?usp=sharing)

### Integrantes del grupo

Simón Di Leo <br>
Kevin Monti <br>
Giménez Matías <br>
Federico Simone

## Introducción

Una blockchain es una base de datos distribuida donde un conjunto de nodos interactúan en modo descentralizado (p2p) para almacenar un conjunto de registros coherentes entre cada uno de los nodos.
La coherencia de dicha información, en una arquitectura PoW (Proof of Work), está garantizada por un proceso denominado minería, que debido a su complejidad, usualmente se ejecuta sobre la GPU.
Afortunadamente para usted, a lo largo de esta cursada y de esta guía aprenderá a crear una red desde cero y a programar en CUDA los algoritmos de minería.

Para apoyar los desarrollos de la blockchain, usted utilizará servicios de un sistema distribuido. Esta, es la respuesta a la necesidad de tener escalabilidad horizontal.

-   Es inherente a existencia de un sistema distribuido la existencia de 2 o más nodos.
-   En el núcleo de un sistema distribuido se encuentra el procesamiento asincrónico.

### Estructura de la blockchain

El concepto básico de blockchain es bastante simple: una base de datos que mantiene una lista en continuo crecimiento de registros ordenados. Algo muy similar a un log de transacciones de una base de datos.

![image](https://github.com/SimonDiLeoGIT/SDyPP-TPIntegrador/assets/117539520/91822796-9a0d-4a56-a893-56268c2aa71b)

Como se puede observar en la imagen, existe un orden y una secuencialidad en las operaciones que se registran en una blockchain, haciendo que, si bien el contenido de cada bloque se puede generar de forma distribuida, su procesamiento debe ser centralizado.

El objetivo de este proyecto, es presentar un prototipo de arquitectura que permita paralelizar y distribuir la generación de bloques (blockchain paralelizable), gráficamente sería algo así:

![image](https://github.com/SimonDiLeoGIT/SDyPP-TPIntegrador/assets/117539520/533b08f5-608d-4d23-9cd9-564eb1bade40)

La principal ventaja de esta arquitectura es que, si dos operaciones no son mutuamente excluyentes o secuenciales, pueden ser realizadas en paralelo.

Para lograr esto, se propone utilizar herramientas vistas en la materia como RabbitMQ para el manejo de colas de los bloques a procesar, Redis como motor de base de datos para registrar los bloques y transacciones y CUDA para el cálculo criptográfico intensivo de hashes y resolución de desafíos. Por último, desarrollar un servidor (coordinador) para la comunicación entre todos las tareas.

## Diagrama de arquitectura ([Miro](https://miro.com/app/board/uXjVK_AxiMg=/?share_link_id=143461907782))

![blockchain](https://github.com/SimonDiLeoGIT/SDyPP-TPIntegrador/assets/117539520/250f3333-904c-4b1c-8c07-1c64f9ddb9e9)

## Parte 1

Todos los hits de la parte 1 (hits del 1 al 8) se encuentran en la carpeta HITS_cuda, respondiendo las preguntas teoricas, implementando el codigo necesario, y documentando como compilar y ejecutar el mismo.

## Parte 2

El despliegue de esta parte se explica en INSTRUCTIONS.md
