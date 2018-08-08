FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    # install python 2
    python \
    python-dev \
    python-pip \
    python-virtualenv \
    python-wheel \
    pkg-config \
    python-nose \
    # requirements for numpy
    libopenblas-base \
    python-numpy \
    python-scipy \
    python-pandas \
    # requirements for keras
    python-h5py \
    python-yaml \
    python-pydot \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


# updating packages
RUN pip install --upgrade setuptools \
    pip \
    jupyter \
    matplotlib

RUN pip install -U tensorflow

ENV KERAS_BACKEND=tensorflow
RUN git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git

# quick test and dump package lists
RUN python -c "import tensorflow as tf; print(tf.__version__)" \
 && dpkg-query -l > /dpkg-query-l.txt \
 && pip2 freeze > /pip2-freeze.txt

WORKDIR /srv/
