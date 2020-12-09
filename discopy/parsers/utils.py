from .pipeline import ParserPipeline
from ..components.argument.base import ExplicitArgumentExtractor, ImplicitArgumentExtractor
from ..components.argument.gosh import GoshArgumentExtractor
from ..components.connective.base import ConnectiveClassifier
from ..components.sense.explicit.base import ExplicitSenseClassifier
from ..components.sense.implicit.base import NonExplicitSenseClassifier


def get_parser(cls_str: str) -> ParserPipeline:
    if cls_str in available_parsers:
        return available_parsers.get(cls_str)
    else:
        raise ValueError('Parser not available')


LinParser = ParserPipeline([
    ConnectiveClassifier(),
    ExplicitArgumentExtractor(),
    ExplicitSenseClassifier(),
    ImplicitArgumentExtractor(),
    NonExplicitSenseClassifier()
])

LinArgumentParser = ParserPipeline([
    ConnectiveClassifier(),
    ExplicitArgumentExtractor(),
])

GoshParser = ParserPipeline([
    ConnectiveClassifier(),
    GoshArgumentExtractor(),
    ExplicitSenseClassifier(),
    ImplicitArgumentExtractor(),
    NonExplicitSenseClassifier()
])

available_parsers = {
    'lin': LinParser,
    'lin-arg': LinArgumentParser,
    'gosh': GoshParser
}
