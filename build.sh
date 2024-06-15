docker-compose down
docker image rm blocks-coordinator:latest
docker build -t blocks-coordinator:latest ./blocks-coordinator/
docker-compose up -d