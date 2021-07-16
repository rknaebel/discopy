# Shallow Discourse Parser
This project aims to provide an implementation of the standard Lin et al. architecture as well as recent advances in neural architectures.
It consists of a parser pipeline architecture which stacks individual parser components to continuously add discourse information.
The focus is currently on explicit relations that were handled first in most pipelines.
Further, remaining sentence pairs without explicit sense relation are processed with the non-explicit component.
The current implementation is following the Conll2016 implementation guidelines.
It aaccepts PDTB2 CoNLL format as input for training and evaluation and mainly produces a line-based json document format.

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

## Usage
*Discopy* currently supports different modes and distinguishes standard models and transformer based models.
These example commands are executed from within the repository folder.

### Evaluation

```shell script
python cli/eval.py lin path/to/model path/to/conll
```

### Standard Architecture

#### Training
```shell script
python cli/train.py lin path/to/model path/to/conll
```
Training data format is json, the folder contains subfolders `en.{train,dev,test}` with files `relations.json` and `parses.json`.


#### Prediction
```shell script
python cli/predict.py -i path/to/some/textfile -m models/lin
```
```shell script
python cli/parse.py -i path/to/some/textfile -m models/lin
```

### Neural Architecture
Neural components are a little bit more complex and ofter require/allow for more hyper-parameters while designing the 
component and throughout the training process.
The training cli gives only a single component-parameter choice.
For individual adaptions, one has to write its own training script.
 
#### Training
```shell script
python cli/bert/train.py [BERT-MODEL] [MODEL-PATH] [CONLL-PATH]
```
Training data format follows the one above.

#### Prediction
```shell script
python cli/bert/predict.py [BERT-MODEL] [MODEL-PATH] [CONLL-PATH]
```
```shell script
python cli/bert/parse.py [BERT-MODEL] [MODEL-PATH] --src [JSON-INPUT]
```
