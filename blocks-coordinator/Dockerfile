FROM python:3.9.19-slim-bullseye

# Crear un usuario no root
RUN useradd -m -d /home/nonroot -s /bin/bash nonroot

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y

# Copiar archivos y directorios
COPY ./requirements.txt requirements.txt
COPY ./src/server.py server.py
COPY ./src/model model
COPY ./src/plugins plugins

# Cambiar la propiedad de los archivos al usuario no root
RUN chown -R nonroot:nonroot /app

# Cambiar al usuario no root
USER nonroot

# Instalar las dependencias de Python
RUN pip install -r requirements.txt

EXPOSE 5000

# CMD ["python", "-m", "flask", "--app", "server", "run", "--host", "0.0.0.0", "--debug"]
CMD ["python", "-m", "flask", "--app", "server", "run", "--host", "0.0.0.0"]