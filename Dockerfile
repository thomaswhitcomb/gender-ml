#FROM tensorflow/tensorflow:latest-gpu-py3
FROM ubuntu
RUN apt update && apt upgrade -y
RUN apt-get install -y locales-all python3-pip python3-dev python3-virtualenv
RUN apt install -y curl wget
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y sudo
RUN apt install -y net-tools
RUN apt install -y octave

RUN pip3 install --upgrade pip

RUN pip3 install tensorflow
RUN pip3 install tf-nightly
RUN pip3 install -U "tensorflow==2.0.0-beta1"
RUN pip3 install keras
RUN pip3 install Pandas
RUN pip3 install NumPy
RUN pip3 install matplotlib
RUN pip3 install tensorflow_hub
RUN pip3 install Pillow
RUN pip3 install scipy

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

#RUN python -c "import tensorflow as tf; print('tf version:',tf.__version__,'keras version:',tf.keras.__version__)"
