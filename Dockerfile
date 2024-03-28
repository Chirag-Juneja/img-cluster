FROM ubuntu:latest

ARG JOBS=8

# Install Dependencies
RUN apt-get update && apt-get install -y git python3.6 python3-pip 
RUN python3 -m pip install cmake

# Install onnxruntime
RUN cd /tmp && \
    git clone --depth 1 \
    --branch v1.12.1 https://github.com/Microsoft/onnxruntime.git && \
    cd onnxruntime && \
    ./build.sh --config RelWithDebInfo \
    --build_shared_lib --parallel ${JOBS} && \
    cd build/Linux/RelWithDebInfo/ && \
    make install && \
    rm -rf /tmp/onnxruntime

