FROM ubuntu
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
ENV PYTHONUNBUFFERED=1
ENV DRIVER_BRANCH=535
ENV LINUX_FLAVOUR=generic

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
USER root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install build-essential checkinstall -y
RUN apt-get install python3-dev python3-pip gcc \
    gfortran musl-dev \
    libffi-dev libssl-dev -y
RUN apt-get install libxrender1
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nvidia-driver-535 -y
RUN apt-get install nvidia-dkms-535 -y
RUN apt-get install nvidia-kernel-common-535 
RUN apt-get install linux-modules-nvidia-535-6.8.0-58-generic linux-modules-nvidia-535-generic -y
RUN apt-get install linux-headers-generic -y
# RUN apt-get install -y nvidia-container-toolkit
# install drivers
# sudo apt-get install nvidia-driver-525 -y
# sudo apt-get install nvidia-dkms-525 -y
# sudo apt-get install nvidia-kernel-common-525 
# sudo apt-get install linux-modules-nvidia-525-6.8.0-58-generic linux-modules-nvidia-525-generic -y

COPY . .
COPY ./rsl_rl .
RUN gcc --version
RUN python3 --version
# RUN nvidia-smi
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt --break-system-packages
RUN pip install setuptools --break-system-packages
RUN cd rsl_rl
RUN pip install . --break-system-packages
RUN cd ..

# Switch to the non-privileged user to run the application.

# Copy the source code into the container.

# Run the application.
# CMD python3 ./panda_train.py