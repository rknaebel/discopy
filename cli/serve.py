import os

from fastapi import FastAPI

# TODO run on gpu raises error: supar sequence length datatype problem
from discopy.data.loaders.raw import load_texts
from discopy.parsers.pipeline import ParserPipeline
from discopy.parsers.utils import get_parser

app = FastAPI()
parser: ParserPipeline = None


@app.on_event("startup")
async def startup_event():
    global parser
    parser_model = os.environ.get('PARSER', 'lin')
    parser = get_parser(parser_model)
    model_path = os.environ.get('MODEL_PATH', 'models/lin')
    parser.load(model_path)


@app.get("/")
def parse_doc(txt: str):
    parsed_text = load_texts([txt])[0]
    doc = parser(parsed_text)
    return doc.to_json()
