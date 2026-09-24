from jmespath import parser
from jmespath.visitor import Options
from jmespath.budget import BudgetLimits
from jmespath.budget import current_context

__version__ = '1.1.0'


def compile(expression):
    return parser.Parser().parse(expression)


def search(expression, data, options=None):
    return parser.Parser().parse(expression).search(data, options=options)
