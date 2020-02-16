from .gosh import GoshParser
from .lin import LinParser, LinArgumentParser
from .parser import AbstractBaseParser

available_parsers = {
    'lin': LinParser,
    'lin-arg': LinArgumentParser,
    'gosh': GoshParser
}


def get_parser(cls_str: str) -> AbstractBaseParser:
    if cls_str in available_parsers:
        return available_parsers.get(cls_str)()
    else:
        raise ValueError('Parser not available')
