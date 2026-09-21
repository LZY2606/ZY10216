"""Configurable resource budgets and cancellation for JMESPath evaluation.

The parser cache stores parsed expressions that can be shared by any number
of concurrent ``search`` calls.  All per-evaluation state therefore lives in
this module, never on an AST node:

* :class:`Budget` accumulates the work consumed by a single ``search`` call.
* :class:`CancellationToken` allows a caller to cooperatively cancel a call.
* ``contextvars`` hold the active state so that nested ``search`` calls made
  from a custom function are charged against the parent budget instead of
  receiving a fresh allowance.

A budget limit is inclusive: a call is allowed to consume exactly ``limit``
units and completes successfully; the request that would consume the
``limit + 1``-st unit raises :class:`jmespath.exceptions.BudgetExceededError`.
"""
import contextvars
import threading

from jmespath import exceptions


class BudgetCategory(object):
    """Names of the independently tracked budget categories."""

    #: One unit every time an AST node is dispatched by the interpreter.
    AST_VISIT = 'ast_visit'
    #: One unit every time a single element of an array (or a single
    #: value of an object projection) is examined.  This covers
    #: projections, filters, slices/flatten materialization and the
    #: internal scans performed by built-in functions (``map``,
    #: ``sort_by``, ``min_by``, ``max_by``, ``avg``, ``sum``, ...).
    ARRAY_ITERATION = 'array_iteration'
    #: One unit per executed value comparison.  This covers comparator
    #: nodes (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``) and the
    #: comparisons performed while sorting or finding ``min``/``max``.
    COMPARISON = 'comparison'
    #: One unit every time a JMESPath function (built-in or custom) is
    #: invoked, charged after argument resolution and before validation.
    FUNCTION_CALL = 'function_call'
    #: One unit per element placed into an intermediate collection created
    #: during evaluation (projected lists, flattened/sliced lists,
    #: multi-selects, ``map``/``sort`` results, object key/value lists ...).
    GENERATED_ELEMENT = 'generated_element'


CATEGORIES = (
    BudgetCategory.AST_VISIT,
    BudgetCategory.ARRAY_ITERATION,
    BudgetCategory.COMPARISON,
    BudgetCategory.FUNCTION_CALL,
    BudgetCategory.GENERATED_ELEMENT,
)


def _validate_limit(limit, name):
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(
            '%s must be a non-negative integer, got: %r' % (name, limit))
    return limit


class CancellationToken(object):
    """A cooperative cancellation signal for a (possibly running) search.

    Token instances are plain thread safe objects.  Call :meth:`cancel` from
    another thread (or from a custom function) to request cancellation; the
    interpreter checks the token before every chargeable unit of work and
    raises :class:`jmespath.exceptions.JMESPathCancelledError`.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


class Budget(object):
    """Mutable consumption counters for a single ``search`` invocation.

    A budget is created from :class:`jmespath.Options` for each top level
    search; nested searches (for example a custom function that itself calls
    :func:`jmespath.search`) reuse the active parent budget.
    """

    def __init__(self, limits=None, total_limit=None, token=None):
        self.limits = {}
        if limits:
            unknown = sorted(set(limits) - set(CATEGORIES))
            if unknown:
                raise ValueError(
                    'Unknown budget categories: %s' % ', '.join(unknown))
            for category in CATEGORIES:
                self.limits[category] = _validate_limit(
                    limits.get(category), 'Budget limit for %r' % category)
        self.total_limit = _validate_limit(total_limit, 'total_limit')
        self.consumed = dict((category, 0) for category in CATEGORIES)
        self.token = token

    @classmethod
    def from_options(cls, options):
        if options is None:
            return cls()
        return cls(limits=options.budget_limits,
                   total_limit=options.total_budget,
                   token=options.cancellation_token)

    @property
    def total_consumed(self):
        return sum(self.consumed.values())

    def remaining(self, category):
        limit = self.limits.get(category)
        if limit is None:
            return None
        return limit - self.consumed[category]

    def snapshot(self):
        return BudgetSnapshot(consumed=dict(self.consumed),
                              total_consumed=self.total_consumed,
                              limits=dict(self.limits),
                              total_limit=self.total_limit)

    def charge(self, category, amount=1):
        """Record ``amount`` units of ``category`` work.

        The cancellation token is consulted first so that a cancellation is
        never hidden by a budget failure.  Limits are then checked using
        inclusive semantics: consuming exactly the limit succeeds.
        """
        token = self.token
        if token is not None and token.cancelled:
            raise exceptions.JMESPathCancelledError(
                locations=_current_locations())
        current = self.consumed[category] + amount
        limit = self.limits.get(category)
        total = self.total_consumed + amount
        if (limit is not None and current > limit) or (
                self.total_limit is not None and total > self.total_limit):
            raise exceptions.BudgetExceededError(
                category=category,
                attempted=current,
                category_limit=limit,
                attempted_total=total,
                total_limit=self.total_limit,
                consumed=dict(self.consumed, **{category: current}),
                locations=_current_locations())
        self.consumed[category] = current


class BudgetSnapshot(object):
    """Immutable after-construction view of a finished evaluation."""

    __slots__ = ('consumed', 'total_consumed', 'limits', 'total_limit')

    def __init__(self, consumed, total_consumed, limits, total_limit):
        self.consumed = consumed
        self.total_consumed = total_consumed
        self.limits = limits
        self.total_limit = total_limit

    def __repr__(self):
        return ('BudgetSnapshot(total_consumed=%d, consumed=%r)' %
                (self.total_consumed, self.consumed))


# ---------------------------------------------------------------------------
# Active evaluation state.  Everything in this section is private to the
# interpreter/function implementation; custom functions interact with it
# through :class:`FunctionContext`.
# ---------------------------------------------------------------------------

_STATE = contextvars.ContextVar('jmespath_budget_state', default=None)


class _State(object):
    def __init__(self, budget, observer):
        self.budget = budget
        self.observer = observer
        # Stack of AST node labels on the current evaluation path.
        self.ast_stack = []
        # Stack of data location segments on the current evaluation path.
        self.data_stack = ['$']
        # Stack of expression frame markers (one per nested ``search``).
        self.expr_stack = []


def current_budget():
    state = _STATE.get()
    if state is None:
        return None
    return state.budget


def _truncate(text, length=80):
    if len(text) > length:
        return text[:length - 3] + '...'
    return text


def _safe_segment(segment):
    text = str(segment)
    return text.replace('~', '~0').replace('/', '~1')


def _render_ast_path(state):
    parts = []
    for label in state.ast_stack:
        if isinstance(label, tuple):
            parts.append('%s[%d]' % (label[0], label[1]))
        elif label.startswith('$'):
            parts.append(label)
        else:
            parts.append(label)
    return '/'.join(parts)


def _render_data_path(state):
    parts = []
    for segment in state.data_stack:
        if segment == '$' or segment == '[*]':
            parts.append(segment)
        elif isinstance(segment, int):
            parts.append('[%d]' % segment)
        else:
            parts.append('/' + _safe_segment(segment))
    return ''.join(parts)


def _current_locations():
    state = _STATE.get()
    if state is None:
        return None
    return {
        'expression': _truncate(state.expr_stack[-1]) if state.expr_stack
        else None,
        'ast_path': _render_ast_path(state),
        'data_path': _render_data_path(state),
    }


def _new_root_state(options):
    return _State(budget=Budget.from_options(options),
                  observer=getattr(options, 'budget_observer', None))


def _run(interpreter, node, value, expression, state, frame):
    """Run ``interpreter`` after ``state``/``frame`` have been prepared."""
    token = _STATE.set(state)
    try:
        state.ast_stack.append(frame)
        state.expr_stack.append(expression)
        try:
            return interpreter.visit(node, value)
        finally:
            state.expr_stack.pop()
            state.ast_stack.pop()
    finally:
        _STATE.reset(token)


def run_interpreter(interpreter, node, value, expression, options=None):
    """Run ``interpreter`` with fresh or inherited budget state."""
    parent = _STATE.get()
    frame = '$%s' % _truncate(expression)
    if parent is not None:
        # A custom function recursively invoked jmespath.search(): the
        # child evaluation is billed to the parent budget.  Limits supplied
        # to the nested Options are intentionally ignored so that a child
        # call cannot reset the parent allowance.  Path stacks are extended
        # so diagnostics show the full call chain.
        state = _State(budget=parent.budget, observer=None)
        state.ast_stack = list(parent.ast_stack)
        state.data_stack = list(parent.data_stack) + [frame]
        return _run(interpreter, node, value, expression, state, frame)
    state = _new_root_state(options)
    try:
        return _run(interpreter, node, value, expression, state, frame)
    finally:
        if state.observer is not None:
            state.observer(state.budget.snapshot())


def run_ad_hoc(interpreter, node, value, options=None):
    """Run an interpreter instantiated outside of ``ParsedResult.search``."""
    state = _new_root_state(options)
    frame = '$<interpreter>'
    try:
        return _run(interpreter, node, value, None, state, frame)
    finally:
        observer = state.observer
        if observer is not None:
            observer(state.budget.snapshot())


class FunctionContext(object):
    """Restricted budget API exposed to custom function implementations.

    Obtain the active context with :func:`jmespath.function_context` inside
    a function invoked during a ``search`` call.  The context deliberately
    exposes no way to reset counters or alter limits.
    """

    def __init__(self, state):
        self._state = state

    def charge(self, category, amount=1):
        """Bill ``amount`` units of ``category`` to the active budget."""
        if category not in CATEGORIES:
            raise ValueError('Unknown budget category: %r' % (category,))
        self._state.budget.charge(category, amount)

    def consumed(self, category=None):
        """Return units consumed in ``category`` (or the total)."""
        if category is None:
            return self._state.budget.total_consumed
        return self._state.budget.consumed[category]

    def remaining(self, category=None):
        """Return remaining units (``None`` means unlimited)."""
        budget = self._state.budget
        if category is None:
            if budget.total_limit is None:
                return None
            return budget.total_limit - budget.total_consumed
        return budget.remaining(category)

    @property
    def cancelled(self):
        token = self._state.budget.token
        return token is not None and token.cancelled

    @property
    def data_path(self):
        return _render_data_path(self._state)

    @property
    def ast_path(self):
        return _render_ast_path(self._state)

    def search(self, expression, data):
        """Evaluate ``expression`` on the parent budget (never a new one)."""
        import jmespath
        return jmespath.search(expression, data)
