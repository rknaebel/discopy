from argparse import ArgumentParser

import uvicorn
from fastapi import FastAPI
from pydantic.main import BaseModel

from discopy.parsers.pipeline import ParserPipeline
from discopy_data.data.loaders.raw import load_texts
from discopy_data.data.update import update_dataset_embeddings

arg_parser = ArgumentParser()
arg_parser.add_argument("--hostname", default="0.0.0.0", type=str, help="REST API hostname")
arg_parser.add_argument("--port", default=8080, type=int, help="REST API port")
arg_parser.add_argument("--model-path", type=str, help="path to trained discourse parser")
arg_parser.add_argument("--bert-model", default='bert-base-cased', type=str, help="bert model name")
arg_parser.add_argument("--reload", action="store_true", help="Reload service on file changes")
args = arg_parser.parse_args()

app = FastAPI()
parser: ParserPipeline = None


@app.on_event("startup")
async def startup_event():
    global parser
    parser = ParserPipeline.from_config(args.model_path)
    parser.load(args.model_path)


@app.get("/api/parser/config")
def get_parser_config():
    configs = []
    for c in parser.components:
        configs.append(c.get_config())
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
    uvicorn.run("app.run_bert:app", host=args.hostname, port=args.port, reload=args.reload)
