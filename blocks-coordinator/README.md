# Nodo coordinador de la blockchain

Nodo coordinador que será responsable de:

-   Definir cómo se estructuran las transacciones en el sistema, cómo se validan y cómo se agregan a la cadena de bloques.
-   Formar los bloques de tareas que deben ser resueltos por los nodos workers.
-   El nodo coordinador puede ajustar la dificultad de los problemas de PoW (Proof of Work) que deben resolver los nodos trabajadores.
-   Responsable del algoritmo de consenso que permita a todos los nodos en la red acordar la validez de los bloques y las transacciones.
-   (deseable) Pool de transacciones: Construir un mecanismo para manejar un pool de transacciones pendientes que los mineros puedan seleccionar para incluir en los bloques siguientes.
