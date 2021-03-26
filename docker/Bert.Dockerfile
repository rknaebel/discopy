FROM python:3.7
MAINTAINER René Knaebel <rene.knaebel@uni-potsdam.de>

# set gpu as unavailable
ENV CUDA_VISIBLE_DEVICES=''

# set the working directory in the container
WORKDIR /discopy

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt
RUN python -m spacy download en
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

# copy the content of the local src directory to the working directory
COPY setup.py .
COPY README.md .
COPY LICENSE .
COPY discopy discopy
COPY cli cli
COPY app app
COPY examples examples
COPY models/bert-pipe-conll3 model
COPY data data


RUN pip install -e /discopy

# command to run on container start
CMD python app/run_bert.py --model-path model --data-path data --port 5000
