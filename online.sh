
cd ./person-web && git pull && cd ..

docker rm -f app
docker image rm app
docker build -t app  -f ./Dockerfilse .
docker run --name app -p 80:5000 -p 443:5000 -d app