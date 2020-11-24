# Shallow Discourse Parser
This is an implementation of the standard Lin et al. architecture.
It consists of four components for classifying explicit discourse relations.
Further, remaining sentence pairs without explicit sense relation are processed with the non-explicit component.
The current implementation is following the Conll2016 implementation guidelines and uses the PDTB2 CoNLL format.

## Setup
You can easily install *discopy* by using pip:
```shell script
pip install git+https://github.com/rknaebel/discopy
```
or you just clone the repository.
The you can either install discopy through pip
```shell script
pip install -e path/to/discopy
```
or you execute `main.py` directly within the repository directory.

## Usage
*Discopy* currently supports different modes.
These example commands are executed from within the repository folder.

### Training
```shell script
python train.py --parser lin --dir models/lin --conll ../conll2016
```

### Prediction
```shell script
python parse.py -i path/to/some/textfile -m models/lin
```

### Evaluation
```shell script
python test.py --parser lin --dir models/lin --conll ../conll2016
```
