gcloud auth configure-docker
docker build -t autonomoeye-api:v0.2 .
docker tag autonomoeye-api:v0.2 gcr.io/capstone2023-378615/autonomoeye-api
docker push gcr.io/capstone2023-378615/autonomoeye-api