gcloud auth configure-docker
docker build -t autonomoeye:v0.3 .
docker tag autonomoeye:v0.3 gcr.io/capstone2023-378615/autonomoeye
docker push gcr.io/capstone2023-378615/autonomoeye