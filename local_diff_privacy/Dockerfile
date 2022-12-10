FROM python:slim
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n ldpfl && \
    conda activate ldpfl && \
    conda install python=3.9 pip \
    && conda install -y keras \
    && conda install -y numpy \
    && conda install -y tensorflow \
    && conda install -y pytorch \
    && conda install -y matplotlib \
    && conda install -y scikit-learn \
    && conda install -y tqdm
COPY app app
COPY data data
RUN echo 'conda activate ldpfl \n\
alias ldpfl="python ./app/ldpfl.py"' >> /root/.bashrc
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["python ./app/ldpfl.py"]
