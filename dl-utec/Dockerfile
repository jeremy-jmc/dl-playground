FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y

WORKDIR /workspace

COPY . .

VOLUME ["/workspace/"]

CMD ["/bin/bash"]

# docker build -t cudita .
# docker run --gpus all -v $(pwd):/workspace/ -it cudita