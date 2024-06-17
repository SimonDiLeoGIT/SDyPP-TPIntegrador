FROM python:3.9.19-slim-bullseye

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y

# Copiar archivos y directorios
COPY ./requirements.txt requirements.txt
COPY ./src/server.py server.py
COPY ./src/utils utils
COPY ./src/plugins plugins
COPY ./src/model model
COPY .env .env

RUN pip install -r requirements.txt

EXPOSE 5005

CMD ["flask", "--app", "server", "run", "--host", "0.0.0.0", "--port", "5005"]

