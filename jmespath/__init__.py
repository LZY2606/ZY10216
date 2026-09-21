from jmespath import parser
from jmespath import budget
from jmespath.visitor import Options
from jmespath.budget import (
    BudgetCategory,
    CancellationToken,
    BudgetSnapshot,
    FunctionContext,
)
from jmespath.exceptions import (
    BudgetExceededError,
    JMESPathCancelledError,
)

__version__ = '1.1.0'


def compile(expression):
    return parser.Parser().parse(expression)


def search(expression, data, options=None):
    return parser.Parser().parse(expression).search(data, options=options)


def function_context():
    """Return the active :class:`FunctionContext`, or ``None``.

    Custom functions called during a ``search`` evaluation can use the
    returned context to bill their own work against the evaluation's
    budget and to observe cancellation requests.
    """
    state = budget._STATE.get()
    if state is None:
        return None
    return FunctionContext(state)
