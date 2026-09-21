import operator

from jmespath import budget as budget_module
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number


def _equals(x, y):
    if _is_special_number_case(x, y):
        return False
    else:
        return x == y


def _is_special_number_case(x, y):
    # We need to special case comparing 0 or 1 to
    # True/False.  While normally comparing any
    # integer other than 0/1 to True/False will always
    # return False.  However 0/1 have this:
    # >>> 0 == True
    # False
    # >>> 0 == False
    # True
    # >>> 1 == True
    # True
    # >>> 1 == False
    # False
    #
    # Also need to consider that:
    # >>> 0 in [True, False]
    # True
    if _is_actual_number(x) and x in (0, 1):
        return isinstance(y, bool)
    elif _is_actual_number(y) and y in (0, 1):
        return isinstance(x, bool)


def _is_comparable(x):
    # The spec doesn't officially support string types yet,
    # but enough people are relying on this behavior that
    # it's been added back.  This should eventually become
    # part of the official spec.
    return _is_actual_number(x) or isinstance(x, string_type)


def _is_actual_number(x):
    # We need to handle python's quirkiness with booleans,
    # specifically:
    #
    # >>> isinstance(False, int)
    # True
    # >>> isinstance(True, int)
    # True
    if isinstance(x, bool):
        return False
    return isinstance(x, Number)


class Options(object):
    """Options to control how a JMESPath function is evaluated."""
    def __init__(self, dict_cls=None, custom_functions=None,
                 budget_limits=None, total_budget=None,
                 cancellation_token=None, budget_observer=None):
        #: The class to use when creating a dict.  The interpreter
        #  may create dictionaries during the evaluation of a JMESPath
        #  expression.  For example, a multi-select hash will
        #  create a dictionary.  By default we use a dict() type.
        #  You can set this value to change what dict type is used.
        #  The most common reason you would change this is if you
        #  want to set a collections.OrderedDict so that you can
        #  have predictable key ordering.
        self.dict_cls = dict_cls
        self.custom_functions = custom_functions
        #: Per-category maximum work units, keyed by
        #: :class:`jmespath.budget.BudgetCategory`.  ``None`` (or a missing
        #: category) means unlimited.  Limits are validated eagerly so that
        #: misconfiguration fails before any evaluation starts.
        self.budget_limits = self._validate_limits(budget_limits)
        #: Maximum total work units summed across all categories.
        self.total_budget = budget_module._validate_limit(
            total_budget, 'total_budget')
        #: Optional :class:`jmespath.budget.CancellationToken` used to
        #: cooperatively cancel an in-progress evaluation.
        self.cancellation_token = cancellation_token
        #: Optional callable invoked once when the (top level) evaluation
        #: finishes (also on failure) with a
        #: :class:`jmespath.budget.BudgetSnapshot`.
        self.budget_observer = budget_observer

    def _validate_limits(self, limits):
        if limits is None:
            return None
        validated = {}
        for category, limit in limits.items():
            if category not in budget_module.CATEGORIES:
                raise ValueError(
                    'Unknown budget category: %r' % (category,))
            validated[category] = budget_module._validate_limit(
                limit, 'Budget limit for %r' % category)
        return validated


class _Expression(object):
    def __init__(self, expression, interpreter, ast_prefix=None,
                 data_prefix=None):
        self.expression = expression
        self.interpreter = interpreter
        # Stack snapshots captured where the expression reference was
        # created; re-entering the expression restores these paths.
        self.ast_prefix = ast_prefix
        self.data_prefix = data_prefix

    def visit(self, node, *args, **kwargs):
        return self.interpreter.visit_expression_reference(
            node, self, *args, **kwargs)


class Visitor(object):
    def __init__(self):
        self._method_cache = {}

    def visit(self, node, *args, **kwargs):
        node_type = node['type']
        method = self._method_cache.get(node_type)
        if method is None:
            method = getattr(
                self, 'visit_%s' % node['type'], self.default_visit)
            self._method_cache[node_type] = method
        return method(node, *args, **kwargs)

    def default_visit(self, node, *args, **kwargs):
        raise NotImplementedError("default_visit")


class TreeInterpreter(Visitor):
    COMPARATOR_FUNC = {
        'eq': _equals,
        'ne': lambda x, y: not _equals(x, y),
        'lt': operator.lt,
        'gt': operator.gt,
        'lte': operator.le,
        'gte': operator.ge
    }
    _EQUALITY_OPS = ['eq', 'ne']
    MAP_TYPE = dict

    def __init__(self, options=None):
        super(TreeInterpreter, self).__init__()
        self._dict_cls = self.MAP_TYPE
        if options is None:
            options = Options()
        self._options = options
        if options.dict_cls is not None:
            self._dict_cls = self._options.dict_cls
        if options.custom_functions is not None:
            self._functions = self._options.custom_functions
        else:
            self._functions = functions.Functions()

    # ------------------------------------------------------------------
    # Path tracking and metering helpers
    # ------------------------------------------------------------------

    @property
    def _state(self):
        return budget_module._STATE.get()

    def _charge(self, category, amount=1):
        state = self._state
        if state is not None:
            state.budget.charge(category, amount)

    def visit(self, node, value):
        # Entry point from budget_module.run_interpreter: a frame marker is
        # already on the stacks, so only the root node label is pushed here.
        state = self._state
        if state is None:
            # An interpreter instantiated directly (internal/backwards
            # compatible usage): provide it with its own budget state.
            return budget_module.run_ad_hoc(self, node, value, self._options)
        return self._dispatch(node, value)

    def _dispatch(self, node, value):
        state = self._state
        state.ast_stack.append(node['type'])
        try:
            self._charge(budget_module.BudgetCategory.AST_VISIT)
            return super(TreeInterpreter, self).visit(node, value)
        finally:
            state.ast_stack.pop()

    def _child(self, node, child, index, value, segment=None):
        state = self._state
        state.ast_stack.append((child['type'], index))
        if segment is not None:
            state.data_stack.append(segment)
        try:
            return self._dispatch(child, value)
        finally:
            if segment is not None:
                state.data_stack.pop()
            state.ast_stack.pop()

    def visit_expression_reference(self, node, expression, value):
        state = self._state
        saved_ast = state.ast_stack
        saved_data = state.data_stack
        state.ast_stack = list(expression.ast_prefix)
        state.data_stack = list(expression.data_prefix) + ['[*]']
        try:
            return self._dispatch(node, value)
        finally:
            state.ast_stack = saved_ast
            state.data_stack = saved_data

    def default_visit(self, node, *args, **kwargs):
        raise NotImplementedError(node['type'])

    def visit_subexpression(self, node, value):
        result = value
        for index, child in enumerate(node['children']):
            result = self._child(node, child, index, result)
        return result

    def visit_field(self, node, value):
        get = getattr(value, 'get', None)
        if get is None:
            return None
        state = self._state
        try:
            present = node['value'] in value
        except TypeError:
            present = False
        if present:
            state.data_stack.append(node['value'])
            try:
                return get(node['value'])
            finally:
                state.data_stack.pop()
        return get(node['value'])

    def visit_comparator(self, node, value):
        # Common case: comparator is == or !=
        comparator_func = self.COMPARATOR_FUNC[node['value']]
        if node['value'] in self._EQUALITY_OPS:
            left = self._child(node, node['children'][0], 0, value)
            right = self._child(node, node['children'][1], 1, value)
            self._charge(budget_module.BudgetCategory.COMPARISON)
            return comparator_func(left, right)
        else:
            # Ordering operators are only valid for numbers.
            # Evaluating any other type with a comparison operator
            # will yield a None value.
            left = self._child(node, node['children'][0], 0, value)
            right = self._child(node, node['children'][1], 1, value)
            if not (_is_comparable(left) and
                    _is_comparable(right)):
                # No comparison is performed on type mismatches.
                return None
            self._charge(budget_module.BudgetCategory.COMPARISON)
            try:
                return comparator_func(left, right)
            except TypeError:
                # Cross-type ordering (e.g. string vs number) yields null
                # rather than an error, matching the historical behavior.
                return None

    def visit_current(self, node, value):
        return value

    def visit_expref(self, node, value):
        state = self._state
        return _Expression(node['children'][0], self,
                           ast_prefix=list(state.ast_stack),
                           data_prefix=list(state.data_stack))

    def visit_function_expression(self, node, value):
        resolved_args = []
        for index, child in enumerate(node['children']):
            current = self._child(node, child, index, value)
            resolved_args.append(current)
        # Charged once per invocation, before validation, so that calls
        # failing with arity/type errors still consume a function unit.
        self._charge(budget_module.BudgetCategory.FUNCTION_CALL)
        return self._functions.call_function(node['value'], resolved_args)

    def visit_filter_projection(self, node, value):
        base = self._child(node, node['children'][0], 0, value)
        if not isinstance(base, list):
            return None
        comparator_node = node['children'][2]
        rhs_node = node['children'][1]
        collected = []
        for index, element in enumerate(base):
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
            if self._is_true(
                    self._child(node, comparator_node, 2, element,
                                segment=index)):
                current = self._child(node, rhs_node, 1, element,
                                      segment=index)
                if current is not None:
                    self._charge(
                        budget_module.BudgetCategory.GENERATED_ELEMENT)
                    collected.append(current)
        return collected

    def visit_flatten(self, node, value):
        base = self._child(node, node['children'][0], 0, value)
        if not isinstance(base, list):
            # Can't flatten the object if it's not a list.
            return None
        merged_list = []
        for index, element in enumerate(base):
            # Walking the outer array is one iteration per element.
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
            if isinstance(element, list):
                state = self._state
                state.data_stack.append(index)
                for sub_index, sub_element in enumerate(element):
                    # Every element pulled out of a nested list is
                    # iterated and materialized into the output.
                    self._charge(
                        budget_module.BudgetCategory.ARRAY_ITERATION)
                    self._charge(
                        budget_module.BudgetCategory.GENERATED_ELEMENT)
                    state.data_stack.append(sub_index)
                    merged_list.append(sub_element)
                    state.data_stack.pop()
                state.data_stack.pop()
            else:
                # A scalar passes straight through and is only generated.
                self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
                merged_list.append(element)
        return merged_list

    def visit_identity(self, node, value):
        return value

    def visit_index(self, node, value):
        # Even though we can index strings, we don't
        # want to support that.
        if not isinstance(value, list):
            return None
        try:
            result = value[node['value']]
        except IndexError:
            return None
        state = self._state
        state.data_stack.append(node['value'])
        try:
            return result
        finally:
            state.data_stack.pop()

    def visit_index_expression(self, node, value):
        result = value
        for index, child in enumerate(node['children']):
            result = self._child(node, child, index, result)
        return result

    def visit_slice(self, node, value):
        if not isinstance(value, list):
            return None
        s = slice(*node['children'])
        sliced = value[s]
        for element in sliced:
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
        return sliced

    def visit_key_val_pair(self, node, value):
        return self._child(node, node['children'][0], 0, value)

    def visit_literal(self, node, value):
        return node['value']

    def visit_multi_select_dict(self, node, value):
        if value is None:
            return None
        collected = self._dict_cls()
        for index, child in enumerate(node['children']):
            collected[child['value']] = self._child(
                node, child, index, value)
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
        return collected

    def visit_multi_select_list(self, node, value):
        if value is None:
            return None
        collected = []
        for index, child in enumerate(node['children']):
            collected.append(self._child(node, child, index, value))
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
        return collected

    def visit_or_expression(self, node, value):
        # Short circuit semantics: the right operand is only visited when
        # the left operand is JMESPath-false, so its node visits are never
        # charged otherwise.
        matched = self._child(node, node['children'][0], 0, value)
        if self._is_false(matched):
            matched = self._child(node, node['children'][1], 1, value)
        return matched

    def visit_and_expression(self, node, value):
        matched = self._child(node, node['children'][0], 0, value)
        if self._is_false(matched):
            return matched
        return self._child(node, node['children'][1], 1, value)

    def visit_not_expression(self, node, value):
        original_result = self._child(node, node['children'][0], 0, value)
        if _is_actual_number(original_result) and original_result == 0:
            # Special case for 0, !0 should be false, not true.
            # 0 is not a special cased integer in jmespath.
            return False
        return not original_result

    def visit_pipe(self, node, value):
        # The pipe forces materialization of the left operand (all its work
        # has already been charged by visiting it), but adds no work units
        # of its own.
        result = value
        for index, child in enumerate(node['children']):
            result = self._child(node, child, index, result)
        return result

    def visit_projection(self, node, value):
        base = self._child(node, node['children'][0], 0, value)
        if not isinstance(base, list):
            return None
        rhs_node = node['children'][1]
        collected = []
        for index, element in enumerate(base):
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
            current = self._child(node, rhs_node, 1, element, segment=index)
            if current is not None:
                self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
                collected.append(current)
        return collected

    def visit_value_projection(self, node, value):
        base = self._child(node, node['children'][0], 0, value)
        try:
            items = list(base.items())
        except AttributeError:
            return None
        rhs_node = node['children'][1]
        collected = []
        for key, element in items:
            # Per-value iteration uses the same category as iterating an
            # array; the data path segment records the object key.
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
            current = self._child(node, rhs_node, 1, element, segment=key)
            if current is not None:
                self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
                collected.append(current)
        return collected

    def _is_false(self, value):
        # This looks weird, but we're explicitly using equality checks
        # because the truth/false values are different between
        # python and jmespath.
        return (value == '' or value == [] or value == {} or value is None or
                value is False)

    def _is_true(self, value):
        return not self._is_false(value)


class GraphvizVisitor(Visitor):
    def __init__(self):
        super(GraphvizVisitor, self).__init__()
        self._lines = []
        self._count = 1

    def visit(self, node, *args, **kwargs):
        self._lines.append('digraph AST {')
        current = '%s%s' % (node['type'], self._count)
        self._count += 1
        self._visit(node, current)
        self._lines.append('}')
        return '\n'.join(self._lines)

    def _visit(self, node, current):
        self._lines.append('%s [label="%s(%s)"]' % (
            current, node['type'], node.get('value', '')))
        for child in node.get('children', []):
            child_name = '%s%s' % (child['type'], self._count)
            self._count += 1
            self._lines.append('  %s -> %s' % (current, child_name))
            self._visit(child, child_name)
