## Docker

## Unofficial Dockerfile for 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
## https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

# NOTE:
# Building the libraries for this repository requires cuda *DURING BUILD PHASE*, therefore:
# - The default-runtime for container should be set to "nvidia" in the deamon.json file. See this: https://github.com/NVIDIA/nvidia-docker/issues/1033
# - For the above to work, the nvidia-container-runtime should be installed in your host. Tested with version 1.14.0-rc.2
# - Make sure NVIDIA's drivers are updated in the host machine. Tested with 525.125.06
ENV DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"


COPY  environment.yml /tmp/environment.yml
WORKDIR /tmp/

RUN conda env create -f environment.yml && \
    conda init bash && exec bash

RUN conda run -n mega_sam python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu126

COPY base /tmp/base/
WORKDIR /tmp/base/

# RUN conda run -n mega_sam python setup.py install

WORKDIR /mega_sam

# This error occurs because there’s a conflict between the threading layer used
# by Intel MKL (Math Kernel Library) and the libgomp library, 
# which is typically used by OpenMP for parallel processing. 
# This often happens when libraries like NumPy or SciPy are used in combination
# with a multithreaded application (e.g., your Docker container or Python environment).
# Solution, set threading layer explicitly! (GNU or INTEL)
ENV MKL_THREADING_LAYER=GNU