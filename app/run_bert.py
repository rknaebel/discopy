import json
import os
from argparse import ArgumentParser

import uvicorn
from fastapi import FastAPI
from pydantic.main import BaseModel

from discopy.components.argument.bert.conn import ConnectiveArgumentExtractor
from discopy.components.sense.explicit.bert_conn_sense import ConnectiveSenseClassifier
from discopy.data.loaders.raw import load_texts
from discopy.data.update import update_dataset_embeddings
from discopy.parsers.pipeline import ParserPipeline

arg_parser = ArgumentParser()
arg_parser.add_argument("--data-path", type=str, help="bert model name")
arg_parser.add_argument("--hostname", default="0.0.0.0", type=str, help="REST API hostname")
arg_parser.add_argument("--port", default=8080, type=int, help="REST API port")
arg_parser.add_argument("--model-path", type=str, help="path to trained discourse parser")
arg_parser.add_argument("--bert-model", default='bert-base-cased', type=str, help="bert model name")
arg_parser.add_argument("--reload", action="store_true", help="Reload service on file changes")
args = arg_parser.parse_args()

app = FastAPI()
parser: ParserPipeline = None
configs = None


@app.on_event("startup")
async def startup_event():
    global data, parser, configs
    configs = json.load(open(os.path.join(args.model_path, 'config.json'), 'r'))
    parser = ParserPipeline([
        ConnectiveSenseClassifier(input_dim=configs[0]['input_dim'], used_context=configs[0]['used_context']),
        ConnectiveArgumentExtractor(window_length=configs[1]['window_length'], input_dim=configs[1]['input_dim'],
                                    hidden_dim=configs[1]['hidden_dim'], rnn_dim=configs[1]['rnn_dim']),
    ])
    parser.load(args.model_path)


@app.get("/api/parser/config")
def get_parser_config():
    return configs


class ParserRequest(BaseModel):
    text: str


@app.post("/api/parser")
def apply_parser(r: ParserRequest):
    docs = load_texts([r.text])
    update_dataset_embeddings(docs, bert_model=args.bert_model)
    doc = parser(docs[0])
    return doc.to_json()


if __name__ == '__main__':
    uvicorn.run("app.run_bert:app", host=args.hostname, port=args.port, log_level="debug", reload=args.reload)
