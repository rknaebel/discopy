FROM python:3.7
MAINTAINER René Knaebel <rene.knaebel@uni-potsdam.de>

# copy parser models to torch cache
#ADD https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.constituency.char.zip /root/.cache/torch/hub/checkpoints/
#ADD https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.char.zip /root/.cache/torch/hub/checkpoints/
ADD models/ptb.crf.constituency.char.zip /root/.cache/torch/hub/checkpoints/
ADD models/ptb.biaffine.dependency.char.zip /root/.cache/torch/hub/checkpoints/

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
COPY ./setup.py /discopy
COPY ./README.md /discopy
COPY ./LICENSE /discopy
COPY ./discopy /discopy/discopy
COPY ./cli /discopy/cli
COPY ./examples /discopy/examples
COPY ./models/lin /discopy/models/lin

RUN pip install -e /discopy

# command to run on container start
CMD [ "uvicorn", "--host", "0.0.0.0", "--port", "8000", "cli.serve:app"]
