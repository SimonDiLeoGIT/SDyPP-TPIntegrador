# Hit #3 - Librerías CUDA

## Visite https://github.com/nvidia/cccl y expanda su informe comentando de que se trata este repositorio, ¿cuándo fue la última vez que se actualizó?

El repositorio "cccl" en GitHub pertenece a NVIDIA y en él unifica tres librerías esenciales para el desarrollo CUDA C++ en un único repositorio. La idea de la libería cccl, es proveer herramientas para ayudar al desarrollador CUDA a escribir código más seguro y eficiente.

Las librerías que se combinan son:

-   Thrust. Librería para realizar operaciones en paralelo usando C++.
-   CUB. Librería de más bajo nivel que nos provee herramientas para desarrollar implementaciones más eficientes en CUDA, de manera tal que se logre maximizar el rendimiento y el uso de la GPU.
-   libcudacxx. Librería que permite escribir código en C++ estándar pero que pueda ejecutarse tanto en la CPU como en la GPU en un entorno CUDA.

Básicamente, este repositorio busca unificar herramientas útiles para el desarrollo cuda. Su última actualización fue hace 42 minutos (siendo hoy, 24/5/2024 17:30hs).

## Visite https://developer.nvidia.com/thrust y documente en su informe de que se trata.

Thrust es una biblioteca de programación paralela de alto nivel diseñada para CUDA. Con Thrust, los programadores de C++ pueden escribir unas pocas líneas de código para realizar operaciones de ordenación, escaneo, transformación y reducción aceleradas en la GPU a una velocidad mucho mayor que la de las CPU más modernas.

Está integrada en el kit de herramientas de CUDA y proporciona una serie de algoritmos y estructuras de datos para facilitar el desarrollo de aplicaciones paralelas en la GPU utilizando el lenguaje de programación C++.

Thrust abstrae los detalles de bajo nivel de la programación en CUDA, permitiendo a los desarrolladores escribir código de manera más sencilla y expresiva, y aprovechar la potencia de la GPU de NVIDIA para realizar cálculos de manera eficiente.

## Compile y ejecute el primer ejemplo que se le presenta en https://docs.nvidia.com/cuda/thrust/index.html#vectors ¿Necesito instalar algo adicional o ya estaba disponible con CUDA?

El primer ejemplo que se presenta en este sitio es un generador de 32 millones de números randoms.
[Random Numbers](random_numbers.cu)

-   Compilación y ejecución:

```sh
nvcc -o random_numbers random_numbers.cu -std=c++11 -arch=sm_35
./random_numbers
```

No es necesario instalar nada adicional ya que Thrust está incluido en el kit de herramientas de CUDA.
Hemos hecho una pequeña modificación en el código para que devuelva los primeros diez elementos generados, y así comprobar que está funcionando.

-   Visite https://docs.nvidia.com/cuda/pdf/Thrust_Quick_Start_Guide.pdf y luego de leerlo y analizarlo comente en su informe cuales son las diferencias entre programar CUDA “a pelo” vs thrust/cccl.

Tras revisar el documento ["Thrust Quick Start Guide" de NVIDIA](https://docs.nvidia.com/cuda/pdf/Thrust_Quick_Start_Guide.pdf), hemos destacado las siguientes diferencias entre programar en CUDA "a pelo" y utilizar Thrust o cccl:

### Abstracción y simplificación

Cuando se programa en CUDA "a pelo", los desarrolladores deben lidiar con detalles de bajo nivel de la arquitectura de GPU, como la gestión de la memoria, la planificación de los hilos y los bloques, y la optimización del rendimiento. Esto puede resultar complejo y requerir adentrarse de manera más profunda a la programación CUDA.

Thrust proporciona una capa de abstracción de alto nivel que simplifica este proceso al ocultar esos detalles de bajo nivel. Por ejemplo, como se puede ver en la sección 2 (vectors) del artículo, Thrust nos facilita la operación de copiar datos desde la memoria de la CPU a la de la GPU a partir del uso del `host_vector` y el `device_vector`. Si bien, por debajo utiliza operaciones con `cudaMemcpy` tal y como lo deberiamos hacer si programaramos "a pelo", nos facilita mucho el proceso durante el desarrollo.

Basicamente, Thrust nos provee de una API que nos permite realizar operaciones comunes como ordenar, buscar, filtrar datos rapmidamente. En lugar de tener que escribir código específico para cada tarea, los desarrolladores pueden utilizar algoritmos predefinidos y estructuras de datos que están optimizados para funcionar de manera eficiente en la GPU.

### Optimización

Cuando se programa en CUDA "a pelo", los desarrolladores tienen un control total sobre cada aspecto del código. Esto significa que pueden ajustar y optimizar cada parte del programa para aprovechar al máximo el rendimiento de la GPU. Pueden utilizar técnicas avanzadas de paralelización, gestionar eficientemente la memoria y optimizar los algoritmos para lograr el mejor rendimiento posible en un escenario específico.

En cambio, Thrust está diseñados para proporcionar una solución general que funcione bien en una amplia variedad de casos de uso. Estas bibliotecas ofrecen abstracciones de alto nivel que simplifican el proceso de desarrollo y permiten a los desarrolladores escribir código de manera más rápida y fácil. Sin embargo, debido a esta abstracción, es posible que no ofrezcan el mismo nivel de optimización que se puede lograr al programar en CUDA "a pelo". Las optimizaciones específicas de la aplicación pueden requerir un conocimiento más profundo de la arquitectura de la GPU y pueden no ser aplicables a todas las situaciones.
