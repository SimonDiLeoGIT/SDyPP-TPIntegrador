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

- Compile y ejecute el primer ejemplo que se le presenta en https://docs.nvidia.com/cuda/thrust/index.html#vectors ¿Necesito instalar algo adicional o ya estaba disponible con CUDA? 

## Primer ejemplo

El primer ejemplo que se presenta en este sitio es un generador de 32 millones de números randoms.
[Random Numbers](/PARTE%201/HIT3/random_numbers.cu)
No es necesario instalar nada adicional ya que Thrust está incluido en el kit de herramientas de CUDA.
Hemos hecho una pequeña modificación en el código para que devuelva los primeros diez elementos generados, y así comprobar que está funcionando.

- Visite https://docs.nvidia.com/cuda/pdf/Thrust_Quick_Start_Guide.pdf y luego de leerlo y analizarlo comente en su informe cuales son las diferencias entre programar CUDA “a pelo” vs thrust/cccl.

Tras revisar el documento "Thrust Quick Start Guide" de NVIDIA, hemos destacado las siguientes diferencias entre programar en CUDA "a pelo" y utilizar Thrust o cccl:

### Abstracción y Simplificación
Cuando se programa en CUDA "a pelo", los desarrolladores deben lidiar con detalles de bajo nivel de la arquitectura de GPU, como la gestión de la memoria, la planificación de los hilos y los bloques, y la optimización del rendimiento. Esto puede resultar complejo y requerir un conocimiento profundo de los conceptos de programación paralela en GPU.

Por otro lado, Thrust y cccl proporcionan una capa de abstracción de alto nivel que simplifica este proceso al ocultar esos detalles de bajo nivel. En lugar de tener que escribir código específico para cada tarea, los desarrolladores pueden utilizar algoritmos predefinidos y estructuras de datos que están optimizados para funcionar de manera eficiente en la GPU. Estas bibliotecas proporcionan una API que permite a los desarrolladores realizar operaciones comunes, como ordenar, buscar, filtrar o realizar operaciones de reducción, de manera más simple y rápida.


## Compatibilidad y Portabilidad
Thrust y cccl están diseñados para ser compatibles con una amplia gama de dispositivos CUDA, lo que significa que el código desarrollado utilizando estas bibliotecas funcionará de manera consistente en una variedad de hardware de GPU de NVIDIA. Esto facilita la portabilidad del código entre diferentes plataformas, ya que no es necesario realizar ajustes específicos para cada arquitectura de GPU. Los desarrolladores pueden escribir su código utilizando las abstracciones proporcionadas por Thrust y cccl y confiar en que funcionará de manera similar en diferentes dispositivos CUDA.

Por otro lado, programar en CUDA "a pelo" puede requerir ajustes específicos para cada arquitectura de GPU. Esto se debe a que los detalles de bajo nivel de la arquitectura, como la organización de los bloques y los hilos, pueden variar entre diferentes generaciones de GPU. Por lo tanto, el código escrito directamente en CUDA puede necesitar ser adaptado para funcionar de manera óptima en diferentes dispositivos, lo que puede limitar la portabilidad del código y requerir un esfuerzo adicional por parte del desarrollador.

## Optimización
Cuando se programa en CUDA "a pelo", los desarrolladores tienen un control total sobre cada aspecto del código. Esto significa que pueden ajustar y optimizar cada parte del programa para aprovechar al máximo el rendimiento de la GPU. Pueden utilizar técnicas avanzadas de paralelización, gestionar eficientemente la memoria y optimizar los algoritmos para lograr el mejor rendimiento posible en un escenario específico. Este nivel de control permite maximizar el rendimiento en situaciones donde cada ciclo de CPU cuenta.

Por otro lado, Thrust y cccl están diseñados para proporcionar una solución general que funcione bien en una amplia variedad de casos de uso. Estas bibliotecas ofrecen abstracciones de alto nivel que simplifican el proceso de desarrollo y permiten a los desarrolladores escribir código de manera más rápida y fácil. Sin embargo, debido a esta abstracción, es posible que no ofrezcan el mismo nivel de optimización que se puede lograr al programar en CUDA "a pelo". Las optimizaciones específicas de la aplicación pueden requerir un conocimiento más profundo de la arquitectura de la GPU y pueden no ser aplicables a todas las situaciones.