# Nodo administrador del pool de mineros

Nodo coordinador que será responsable de:

-   Procesar tareas de minería generadas por el nodo coordinador.
-   Generar subtareas para los mineros dividiendo los rangos de búsqueda del nonce
-   Recibir keep-alive de los mineros GPU para conocer el estado del pool de minería
-   Iniciar/destruir instancias de mineros CPU en la nube cuando sea necesario.
