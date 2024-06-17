docker-compose down -v
# docker image rm blocks-coordinator:latest
docker image rm pool-manager:latest
# docker build -t blocks-coordinator:latest ./blocks-coordinator/
docker build -t pool-manager:latest ./pool-manager/
docker-compose up -d
