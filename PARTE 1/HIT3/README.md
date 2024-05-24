# Hit #3 - Librerías CUDA

- Visite https://github.com/nvidia/cccl y expanda su informe comentando de que se trata este repositorio, ¿cuándo fue la última vez que se actualizó?

El repositorio "cccl" en GitHub pertenece a NVIDIA y contiene el código fuente de la biblioteca "CUDA Compatibility Checker Library" (cccl). Esta biblioteca se utiliza para verificar la compatibilidad de los dispositivos CUDA con determinadas características, lo que puede ser útil para garantizar que el código CUDA sea compatible con una amplia gama de dispositivos. La última vez que se actualizó este repositorio fue el 23 de abril de 2024.

- Visite https://developer.nvidia.com/thrust y documente en su informe de que se trata. 

Thrust es una biblioteca de programación paralela de alto nivel diseñada para CUDA. Está integrada en el kit de herramientas de CUDA y proporciona una serie de algoritmos y estructuras de datos para facilitar el desarrollo de aplicaciones paralelas en la GPU utilizando el lenguaje de programación C++. Thrust abstrae los detalles de bajo nivel de la programación en CUDA, permitiendo a los desarrolladores escribir código de manera más sencilla y expresiva, y aprovechar la potencia de la GPU de NVIDIA para realizar cálculos de manera eficiente.

- Compile y ejecute el primer ejemplo que se le presenta en https://docs.nvidia.com/cuda/thrust/index.html#vectors ¿Necesito instalar algo adicional o ya estaba disponible con CUDA? 

## Primer ejemplo

El primer ejemplo que se presenta en este sitio es un generador de 32 millones de números randoms.
[Random Numbers](random_numbers.cu)

- Compilación y ejecución:
```sh
nvcc -o random_number random_number.cu -std=c++11 -arch=sm_35
./random_number
```
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