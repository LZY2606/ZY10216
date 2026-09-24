"""Evaluation budgets and cancellation for JMESPath evaluation.

This module implements an opt-in resource accounting layer.  A budget
tracks five categories of work performed while evaluating an expression:

* ``ast_nodes`` - every evaluation of an AST node.  Nodes that are
  evaluated repeatedly (for example the right hand side of a projection,
  once per input element) are counted once per evaluation.
* ``elements`` - every iteration over an element of a data array
  (projections, filter projections, flatten, ``map()``).
* ``comparisons`` - every evaluation of a comparison expression
  (``==``, ``<``, ...) and every sort/min/max key evaluation performed
  by ``sort_by()``/``min_by()``/``max_by()``.
* ``function_calls`` - every invocation of a JMESPath function.
* ``output_elements`` - every element appended to an intermediate
  collection built during evaluation (projection results, flatten
  output, multi-select lists/dicts, slices, ``map()``, ``sort()``,
  ``sort_by()``, ``reverse()``, ``keys()``, ``values()``).

A limit of ``None`` (the default) means "unlimited" for that category.
The endpoint semantics are fixed: evaluation may consume up to *and
including* the configured limit.  Only consumption strictly greater
than a limit raises :class:`jmespath.exceptions.BudgetExceededError`.

Budget state is created fresh for every ``search()`` call, so ASTs
shared through the parser cache can be evaluated concurrently with
different budgets.
"""
import contextvars

from jmespath import exceptions

CATEGORY_AST_NODES = 'ast_nodes'
CATEGORY_ELEMENTS = 'elements'
CATEGORY_COMPARISONS = 'comparisons'
CATEGORY_FUNCTION_CALLS = 'function_calls'
CATEGORY_OUTPUT_ELEMENTS = 'output_elements'

CATEGORIES = (
    CATEGORY_AST_NODES,
    CATEGORY_ELEMENTS,
    CATEGORY_COMPARISONS,
    CATEGORY_FUNCTION_CALLS,
    CATEGORY_OUTPUT_ELEMENTS,
)

_ACTIVE_CONTEXT = contextvars.ContextVar(
    'jmespath_evaluation_context', default=None)


def current_context():
    """Return the EvaluationContext of the in-progress evaluation.

    Returns the context of the most recent evaluation started on this
    thread (or async task), or ``None`` if no budgeted evaluation has
    run.  Custom functions use this to debit their own work, to cancel
    the evaluation, or to run nested searches that share the parent
    budget.
    """
    return _ACTIVE_CONTEXT.get()


class BudgetLimits(object):
    """Declarative budget limits passed via ``Options(budget=...)``.

    ``total`` caps the sum of consumption across all categories; the
    remaining keyword arguments cap their individual category.  All
    limits are optional.  ``should_cancel`` is an optional callable
    polled during evaluation; when it returns a truthy value the
    evaluation aborts with ``EvaluationCancelledError``.
    """
    def __init__(self, total=None, ast_nodes=None, elements=None,
                 comparisons=None, function_calls=None,
                 output_elements=None, should_cancel=None):
        self.limits = {
            CATEGORY_AST_NODES: ast_nodes,
            CATEGORY_ELEMENTS: elements,
            CATEGORY_COMPARISONS: comparisons,
            CATEGORY_FUNCTION_CALLS: function_calls,
            CATEGORY_OUTPUT_ELEMENTS: output_elements,
        }
        for category, limit in self.limits.items():
            self._validate_limit(category, limit)
        self._validate_limit('total', total)
        self.total = total
        if should_cancel is not None and not callable(should_cancel):
            raise ValueError('should_cancel must be callable or None')
        self.should_cancel = should_cancel

    def _validate_limit(self, name, limit):
        if limit is None:
            return
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError(
                'Budget limit for %r must be an int or None, got: %r' % (
                    name, limit))
        if limit < 0:
            raise ValueError(
                'Budget limit for %r must be >= 0, got: %r' % (
                    name, limit))

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(
            'budget must be a BudgetLimits instance or a dict of '
            'limits, got: %r' % (value,))


class EvaluationContext(object):
    """Per-search budget state, cancellation, and nested search.

    An instance is created by the interpreter for every ``search()``
    call that has a budget configured.  It is never shared between
    independent ``search()`` calls, which is what allows cached ASTs
    to be evaluated concurrently under different budgets.
    """
    def __init__(self, limits, options=None):
        self._limits = limits
        self._options = options
        self._consumed = dict((category, 0) for category in CATEGORIES)
        self._total = 0
        self._cancel_reason = None
        #: The expression being evaluated (filled in by the caller).
        self.expression = None
        #: Best-effort evaluation position, maintained by the
        #: interpreter.  Exposed as plain lists of path segments.
        self.ast_path = []
        self.data_path = []

    @property
    def limits(self):
        return dict(self._limits.limits)

    @property
    def total_limit(self):
        return self._limits.total

    @property
    def consumed(self):
        return dict(self._consumed)

    @property
    def total_consumed(self):
        return self._total

    @property
    def cancelled(self):
        return self._cancel_reason is not None

    def cancel(self, reason=None):
        """Cancel the evaluation.

        The next budget checkpoint raises
        :class:`jmespath.exceptions.EvaluationCancelledError`.
        """
        self._cancel_reason = reason if reason is not None else \
            'evaluation cancelled'

    def consume(self, category, amount=1):
        """Debit ``amount`` of work against ``category`` and the total.

        Raises ``EvaluationCancelledError`` if the evaluation has been
        cancelled, and ``BudgetExceededError`` if a configured limit
        would be exceeded.  Consuming exactly up to a limit is allowed.
        """
        if category not in self._consumed:
            raise ValueError('Unknown budget category: %r' % (category,))
        if amount < 0:
            raise ValueError('amount must be >= 0, got: %r' % (amount,))
        self._check_cancelled()
        if amount == 0:
            return
        self._consumed[category] += amount
        self._total += amount
        limit = self._limits.limits[category]
        if limit is not None and self._consumed[category] > limit:
            raise exceptions.BudgetExceededError(
                category, self._consumed[category], limit,
                diagnostics=self._diagnostics())
        if self._limits.total is not None and \
                self._total > self._limits.total:
            raise exceptions.BudgetExceededError(
                'total', self._total, self._limits.total,
                diagnostics=self._diagnostics())

    def _check_cancelled(self):
        if self._cancel_reason is not None:
            raise exceptions.EvaluationCancelledError(
                self._cancel_reason, diagnostics=self._diagnostics())
        should_cancel = self._limits.should_cancel
        if should_cancel is not None and should_cancel():
            self.cancel('cancelled by should_cancel callback')
            raise exceptions.EvaluationCancelledError(
                self._cancel_reason, diagnostics=self._diagnostics())

    def search(self, expression, data):
        """Evaluate ``expression`` against ``data`` under this budget.

        Nested evaluations debit the *same* budget, so recursive use of
        JMESPath from a custom function cannot bypass the parent
        evaluation's limits.
        """
        from jmespath import parser as parser_module
        from jmespath import visitor as visitor_module
        parsed = parser_module.Parser().parse(expression)
        interpreter = visitor_module.TreeInterpreter(self._options)
        interpreter.context = self
        return interpreter.visit(parsed.parsed, data)

    def _diagnostics(self):
        return {
            'expression': self.expression,
            'ast_path': self._render_ast_path(),
            'data_path': self._render_data_path(),
            'consumed': dict(self._consumed),
            'total_consumed': self._total,
        }

    def _render_ast_path(self):
        if not self.ast_path:
            return 'root'
        return 'root -> ' + ' -> '.join(str(p) for p in self.ast_path)

    def _render_data_path(self):
        if not self.data_path:
            return '$'
        return '$' + ''.join(str(p) for p in self.data_path)
