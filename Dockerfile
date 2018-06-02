FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER Eric Wong <ericwong@cs.cmu.edu>

# set home location
ENV HOME=/home
ENV TERM=xterm-256color

RUN apt-get update && \
    apt-get install -y curl git zsh vim wget zip

# setup environment
RUN git clone --recursive https://github.com/riceric22/dotfiles.git $HOME/.dotfiles && \
    cd $HOME/.dotfiles && \
    ln -s $HOME/.dotfiles/zshrc $HOME/.zshrc && \
    ln -s $HOME/.dotfiles/vimrc $HOME/.vimrc && \
    ln -s $HOME/.dotfiles/vim $HOME/.vim && \
    git clone git://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh \
      && chsh -s /bin/zsh

# install and add conda to path
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh && \
    /bin/bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p /opt/conda

ENV PATH /opt/conda/bin:$PATH

RUN conda update -n base conda && \
    conda install pytorch=0.4 torchvision -c pytorch -y

RUN pip install --upgrade pip && \
    pip install setproctitle line_profiler setGPU waitGPU dotfiles

RUN echo cd >> $HOME/.bashrc

ENV PYTHONPATH /home/:$PATH
ENV CUDA_DEVICE_ORDER PCI_BUS_ID
ENV LANG C.UTF-8 
ENV LC_ALL C.UTF-8

# MNIST data
COPY . /home/
#COPY raw /home/raw/
#COPY processed /home/processed/
# add files from repo
#COPY convex_adversarial /home/convex_adversarial/
#COPY examples /home/examples/