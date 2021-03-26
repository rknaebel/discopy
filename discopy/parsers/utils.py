import discopy.parsers.pipeline
import discopy.components.argument.base
import discopy.components.argument.bert.conn
import discopy.components.argument.bert.explicit
import discopy.components.argument.crf
import discopy.components.argument.gosh
import discopy.components.connective.base
import discopy.components.connective.bert
import discopy.components.sense.explicit.base
import discopy.components.sense.explicit.bert_conn_sense
import discopy.components.sense.implicit.base

components = [
    discopy.components.argument.base.ExplicitArgumentExtractor,
    discopy.components.argument.base.ImplicitArgumentExtractor,
    discopy.components.argument.bert.conn.ConnectiveArgumentExtractor,
    discopy.components.argument.bert.explicit.ExplicitArgumentExtractor,
    discopy.components.argument.crf.CRFArgumentExtractor,
    discopy.components.argument.gosh.GoshArgumentExtractor,
    discopy.components.connective.base.ConnectiveClassifier,
    discopy.components.connective.bert.ConnectiveClassifier,
    discopy.components.sense.explicit.base.ExplicitSenseClassifier,
    discopy.components.sense.explicit.bert_conn_sense.ConnectiveSenseClassifier,
    discopy.components.sense.implicit.base.NonExplicitSenseClassifier,
]

component_register = {}
for comp_cls in components:
    if comp_cls.model_name in component_register:
        raise ValueError("Component name already used.")
    component_register[comp_cls.model_name] = comp_cls


def get_parser(cls_str: str) -> discopy.parsers.pipeline.ParserPipeline:
    if cls_str in available_parsers:
        return available_parsers.get(cls_str)
    else:
        raise ValueError('Parser not available')


LinParser = discopy.parsers.pipeline.ParserPipeline([
    discopy.components.connective.base.ConnectiveClassifier(),
    discopy.components.argument.base.ExplicitArgumentExtractor(),
    discopy.components.sense.explicit.base.ExplicitSenseClassifier(),
    discopy.components.argument.base.ImplicitArgumentExtractor(),
    discopy.components.sense.implicit.base.NonExplicitSenseClassifier()
])

LinArgumentParser = discopy.parsers.pipeline.ParserPipeline([
    discopy.components.connective.base.ConnectiveClassifier(),
    discopy.components.argument.base.ExplicitArgumentExtractor(),
])

GoshParser = discopy.parsers.pipeline.ParserPipeline([
    discopy.components.connective.base.ConnectiveClassifier(),
    discopy.components.argument.gosh.GoshArgumentExtractor(),
    discopy.components.sense.explicit.base.ExplicitSenseClassifier(),
    discopy.components.argument.base.ImplicitArgumentExtractor(),
    discopy.components.sense.implicit.base.NonExplicitSenseClassifier()
])

available_parsers = {
    'lin': LinParser,
    'lin-arg': LinArgumentParser,
    'gosh': GoshParser,
}
