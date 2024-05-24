# Hit #2 - Hola mundo en CUDA

Visite la presentación original de CUDA, o su versión actualizada y realice un programa básico en CUDA de hola mundo como los del ejemplo que se menciona. Comience a elaborar un informe de las tareas realizadas, describiendo:
Qué entorno está utilizando,
Si se encontró con problemas como los solucionó
Cuál es su setup y 
Si usa hardware nativo, las características del mismo.

Los ejemplos a continuación y esta guía fueron probadas con una GPU GT 750M y CUDA 10.1, con data de 2013.
Nótese que entre versiones de CUDA puede tener que realizar cambios en el código y no todos los ejemplos de CUDA 10.1 son compatibles con las versiones más nuevas. Documente las adaptaciones que realice si corresponde.
Consideración: Corto y conciso. 

# Informe de Tareas Realizadas

## Entorno Utilizado
- Conexión: SSH
- Sistema Operativo del Host: Linux
- GPU Utilizada: Tesla K40c
- Controladores de NVIDIA: Versión 470.82.01
- Versión de CUDA: 11.4
- Compilador de CUDA: nvcc

## Problemas Encontrados y Soluciones
- Problema: El programa no imprimía "Hola, mundo" desde la GPU.
- Solución: Añadir verificaciones de errores después de las llamadas importantes de CUDA.
- Problema: Error "no kernel image is available for execution on the device".
- Solución: Especificar la arquitectura correcta de la GPU (Kepler) utilizando la opción -arch=sm_35 al compilar el código con nvcc.

## Setup Utilizado
- Compilador: nvcc

## Comandos de compilación y ejecución:

```sh
nvcc -arch=sm_35 -o hola_mundo hola_mundo.cu
./hola_mundo
```

## Características del Hardware Nativo Utilizado
- GPU: Tesla K40c
- Memoria: 12 GB GDDR5
- Arquitectura de GPU: Kepler (sm_35)
- Capacidad de Cálculo: Compute 3.5

## Adaptaciones Realizadas
- Especificar la arquitectura de la GPU al compilar (-arch=sm_35).
- Añadir verificaciones de errores de CUDA para asegurar una ejecución correcta del kernel y detectar cualquier problema de manera eficiente.

# Hit #3 - Librerías CUDA

- Visite https://github.com/nvidia/cccl y expanda su informe comentando de que se trata este repositorio, ¿cuándo fue la última vez que se actualizó?

El repositorio "cccl" en GitHub pertenece a NVIDIA y contiene el código fuente de la biblioteca "CUDA Compatibility Checker Library" (cccl). Esta biblioteca se utiliza para verificar la compatibilidad de los dispositivos CUDA con determinadas características, lo que puede ser útil para garantizar que el código CUDA sea compatible con una amplia gama de dispositivos. La última vez que se actualizó este repositorio fue el 23 de abril de 2024.

- Visite https://developer.nvidia.com/thrust y documente en su informe de que se trata. 

Thrust es una biblioteca de programación paralela de alto nivel diseñada para CUDA. Está integrada en el kit de herramientas de CUDA y proporciona una serie de algoritmos y estructuras de datos para facilitar el desarrollo de aplicaciones paralelas en la GPU utilizando el lenguaje de programación C++. Thrust abstrae los detalles de bajo nivel de la programación en CUDA, permitiendo a los desarrolladores escribir código de manera más sencilla y expresiva, y aprovechar la potencia de la GPU de NVIDIA para realizar cálculos de manera eficiente.

Compile y ejecute el primer ejemplo que se le presenta en https://docs.nvidia.com/cuda/thrust/index.html#vectors ¿Necesito instalar algo adicional o ya estaba disponible con CUDA? 



Visite https://docs.nvidia.com/cuda/pdf/Thrust_Quick_Start_Guide.pdf y luego de leerlo y analizarlo comente en su informe cuales son las diferencias entre programar CUDA “a pelo” vs thrust/cccl.
Consideración: Corto y conciso.