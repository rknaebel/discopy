# Shallow Discourse Parser
This project aims to provide an implementation of the standard Lin et al. architecture as well as recent advances in neural architectures.
It consists of a parser pipeline architecture which stacks individual parser components to continuously add discourse information.
The focus is currently on explicit relations that were handled first in most pipelines.
Further, remaining sentence pairs without explicit sense relation are processed with the non-explicit component.
The current implementation is following the Conll2016 implementation guidelines.
It accepts PDTB2 CoNLL format as input for training and evaluation and mainly produces a line-based json document format.

The parser is presented at the CODI 2021 Workshop. For more information, checkout the paper 
[discopy: A Neural System for Shallow Discourse Parsing](https://aclanthology.org/2021.codi-main.12/).

## Setup
You can easily install *discopy* by using pip:
```shell script
pip install git+https://github.com/rknaebel/discopy
```
or you just clone the repository.
Then you can install discopy through pip
```shell script
pip install -e path/to/discopy
```

## Usage
*Discopy* currently supports different modes and distinguishes standard feature-based models and neural-based (transformer) models.
These example commands are executed from within the repository folder.

### Evaluation

```shell script
discopy-eval path/to/conll-gold path/to/prediction
```

### Standard Architecture

#### Training
```shell script
discopy-train lin path/to/model path/to/conll
```
Training data format is json, the folder contains subfolders `en.{train,dev,test}` with files `relations.json` and `parses.json`.


#### Prediction
```shell script
discopy-predict lin path/to/conll/en.part path/to/model/lin
```
```shell script
discopy-parse lin path/to/model/lin -i path/to/some/documents.json
```
```shell script
discopy-tokenize -i path/to/textfile | discopy-add-parses -c | discopy-parse lin models/lin
```


### Neural Architecture
Neural components are a little bit more complex and ofter require/allow for more hyper-parameters while designing the 
component and throughout the training process.
The training cli gives only a single component-parameter choice.
For individual adaptions, one has to write its own training script.
The `bert-model` parameter corresponds to the huggingface transformers model names.

#### Training
```shell script
discopy-nn-train [BERT-MODEL] [MODEL-PATH] [CONLL-PATH]
```
Training data format follows the one above.

#### Prediction
```shell script
discopy-nn-predict [BERT-MODEL] [MODEL-PATH] [CONLL-PATH]
```
```shell script
discopy-nn-parse [BERT-MODEL] [MODEL-PATH] -i [JSON-INPUT]
```
```shell script
cat path/to/textfile | discopy-nn-parse [BERT-MODEL] [MODEL-PATH]
```
```shell script
discopy-tokenize --tokenize-only -i path/to/textfile | discopy-nn-parse bert-base-cased models/pipeline-bert-2
```
