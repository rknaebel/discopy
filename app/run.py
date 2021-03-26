from argparse import ArgumentParser

import uvicorn
from fastapi import FastAPI

# TODO run on gpu raises error: supar sequence length datatype problem
from discopy.data.loaders.raw import load_texts
from discopy.data.update import update_dataset_parses
from discopy.parsers.pipeline import ParserPipeline
from discopy.parsers.utils import get_parser

arg_parser = ArgumentParser()
arg_parser.add_argument("--hostname", default="0.0.0.0", type=str, help="REST API hostname")
arg_parser.add_argument("--port", default=8080, type=int, help="REST API port")
arg_parser.add_argument("--parser", type=str, help="Used discourse parser")
arg_parser.add_argument("--model-path", type=str, help="path to trained discourse parser")
arg_parser.add_argument("--reload", action="store_true", help="Reload service on file changes")
args = arg_parser.parse_args()

app = FastAPI()
parser: ParserPipeline = None


@app.on_event("startup")
async def startup_event():
    global parser
    parser = get_parser(args.parser)
    parser.load(args.model_path)


@app.get("/api/parser")
def parse_doc(txt: str):
    docs = load_texts([txt])
    update_dataset_parses(docs)
    doc = parser(docs[0])
    return doc.to_json()


if __name__ == '__main__':
    uvicorn.run("run:app", host=args.hostname, port=args.port, log_level="debug", reload=True)
