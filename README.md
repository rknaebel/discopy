# Shallow Discourse Parser
This is an implementation of the standard Lin et al. architecture.
It consists of four components for classifying explicit discourse relations.
Further, remaining sentence pairs without explicit sense relation are processed with the non-explicit component.
The current implementation is following the Conll2016 implementation guidelines and uses the PDTB2 CoNLL format.

## Setup
You can easily install *discopy* by using pip:
```
pip install git+https://github.com/rknaebel/discopy
```
or you just clone the repository.
The you can either install discopy through pip
```
pip install -e path/to/discopy
```
or you execute `main.py` directly within the repository directory.

## Usage
*Discopy* currently supports different modes.
These example commands are executed from within the repository folder.

### Training
```
python3 main.py --mode train --dir tmp --pdtb path/train/relations.json --parses path/train/parses.json
```
For training all components two epochs each:
```
python3 main.py --mode train --dir tmp --pdtb path/train/relations.json --parses path/train/parses.json --epochs 2
```

### Execution
```
python3 main.py --mode run --dir tmp --parses path/dev/parses.json --out output.json
```

### Evaluation
```
python3 main.py --mode eval --pdtb path/dev/relations.json --out output.json
```


### Semi supervised Tri Training
```
python3 self_baseline.py --dir exp_tri --conll /data/discourse/conll2016/ --train --gpu 0
```

## Components


### Connective


### Argument Position


### Argument Extraction


### Explicit Sense


### Implicit Sense