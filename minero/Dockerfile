FROM python:3.9.19-slim-bullseye

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos y directorios
COPY ./requirements.txt requirements.txt
COPY ./src/server.py server.py
COPY ./src/utils utils
COPY ./src/plugins plugins
COPY ./src/model model
COPY .env.prod .env


# Instalar las dependencias de Python
RUN pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=server.py

# Comando para iniciar la aplicación
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]