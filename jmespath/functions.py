import math
import json

from jmespath import budget as budget_module
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods
from numbers import Number



def _is_actual_number(x):
    if isinstance(x, bool):
        return False
    return isinstance(x, Number)


def _equals(x, y):
    # Mirrors the 0/1 vs True/False special casing used by the interpreter.
    if _is_actual_number(x) and x in (0, 1) and isinstance(y, bool):
        return False
    if _is_actual_number(y) and y in (0, 1) and isinstance(x, bool):
        return False
    return x == y

class _MeteredOrderable(object):
    """Wraps an element plus its sort key and bills comparisons.

    ``raw`` is the original element (returned by ``sort_by``); ``value``
    holds the comparison key.  For plain ``sort``/``min``/``max`` the raw
    element and key are the same.
    """

    __slots__ = ('raw', 'value', '_budget')

    def __init__(self, value, budget, raw=None):
        self.raw = value if raw is None else raw
        self.value = value
        self._budget = budget

    def __lt__(self, other):
        self._budget.charge(budget_module.BudgetCategory.COMPARISON)
        return self.value < other.value

    def __gt__(self, other):
        self._budget.charge(budget_module.BudgetCategory.COMPARISON)
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __hash__(self):
        return hash(self.value)


def _metered_best_min(items):
    # Mirrors the stdlib min()/max() comparison counts exactly: one
    # comparison per non-leading element.
    items = iter(items)
    try:
        best = next(items)
    except StopIteration:
        raise ValueError('arg is an empty sequence')
    for item in items:
        if item < best:
            best = item
    return best


def _metered_best_max(items):
    items = iter(items)
    try:
        best = next(items)
    except StopIteration:
        raise ValueError('arg is an empty sequence')
    for item in items:
        if item > best:
            best = item
    return best


# python types -> jmespath types
TYPES_MAP = {
    'bool': 'boolean',
    'list': 'array',
    'dict': 'object',
    'NoneType': 'null',
    'unicode': 'string',
    'str': 'string',
    'float': 'number',
    'int': 'number',
    'long': 'number',
    'OrderedDict': 'object',
    '_Projection': 'array',
    '_Expression': 'expref',
}


# jmespath types -> python types
REVERSE_TYPES_MAP = {
    'boolean': ('bool',),
    'array': ('list', '_Projection'),
    'object': ('dict', 'OrderedDict',),
    'null': ('NoneType',),
    'string': ('unicode', 'str'),
    'number': ('float', 'int', 'long'),
    'expref': ('_Expression',),
}


def signature(*arguments):
    def _record_signature(func):
        func.signature = arguments
        return func
    return _record_signature


class FunctionRegistry(type):
    def __init__(cls, name, bases, attrs):
        cls._populate_function_table()
        super(FunctionRegistry, cls).__init__(name, bases, attrs)

    def _populate_function_table(cls):
        function_table = {}
        # Any method with a @signature decorator that also
        # starts with "_func_" is registered as a function.
        # _func_max_by -> max_by function.
        for name, method in get_methods(cls):
            if not name.startswith('_func_'):
                continue
            signature = getattr(method, 'signature', None)
            if signature is not None:
                function_table[name[6:]] = {
                    'function': method,
                    'signature': signature,
                }
        cls.FUNCTION_TABLE = function_table


class Functions(metaclass=FunctionRegistry):

    FUNCTION_TABLE = {
    }

    @property
    def _active_budget(self):
        return budget_module.current_budget()

    def _charge(self, category, amount=1):
        budget = budget_module.current_budget()
        if budget is not None:
            budget.charge(category, amount)

    def call_function(self, function_name, resolved_args):
        try:
            spec = self.FUNCTION_TABLE[function_name]
        except KeyError:
            raise exceptions.UnknownFunctionError(
                "Unknown function: %s()" % function_name)
        function = spec['function']
        signature = spec['signature']
        self._validate_arguments(resolved_args, signature, function_name)
        return function(self, *resolved_args)

    def _validate_arguments(self, args, signature, function_name):
        if signature and signature[-1].get('variadic'):
            if len(args) < len(signature):
                raise exceptions.VariadictArityError(
                    len(signature), len(args), function_name)
        elif len(args) != len(signature):
            raise exceptions.ArityError(
                len(signature), len(args), function_name)
        return self._type_check(args, signature, function_name)

    def _type_check(self, actual, signature, function_name):
        for i in range(len(signature)):
            allowed_types = signature[i]['types']
            if allowed_types:
                self._type_check_single(actual[i], allowed_types,
                                        function_name)

    def _type_check_single(self, current, types, function_name):
        # Type checking involves checking the top level type,
        # and in the case of arrays, potentially checking the types
        # of each element.
        allowed_types, allowed_subtypes = self._get_allowed_pytypes(types)
        # We're not using isinstance() on purpose.
        # The type model for jmespath does not map
        # 1-1 with python types (booleans are considered
        # integers in python for example).
        actual_typename = type(current).__name__
        if actual_typename not in allowed_types:
            raise exceptions.JMESPathTypeError(
                function_name, current,
                self._convert_to_jmespath_type(actual_typename), types)
        # If we're dealing with a list type, we can have
        # additional restrictions on the type of the list
        # elements (for example a function can require a
        # list of numbers or a list of strings).
        # Arrays are the only types that can have subtypes.
        if allowed_subtypes:
            self._subtype_check(current, allowed_subtypes,
                                types, function_name)

    def _get_allowed_pytypes(self, types):
        allowed_types = []
        allowed_subtypes = []
        for t in types:
            type_ = t.split('-', 1)
            if len(type_) == 2:
                type_, subtype = type_
                allowed_subtypes.append(REVERSE_TYPES_MAP[subtype])
            else:
                type_ = type_[0]
            allowed_types.extend(REVERSE_TYPES_MAP[type_])
        return allowed_types, allowed_subtypes

    def _subtype_check(self, current, allowed_subtypes, types, function_name):
        if len(allowed_subtypes) == 1:
            # The easy case, we know up front what type
            # we need to validate.
            allowed_subtypes = allowed_subtypes[0]
            for element in current:
                self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
                actual_typename = type(element).__name__
                if actual_typename not in allowed_subtypes:
                    raise exceptions.JMESPathTypeError(
                        function_name, element, actual_typename, types)
        elif len(allowed_subtypes) > 1 and current:
            # Dynamic type validation.  Based on the first
            # type we see, we validate that the remaining types
            # match.
            first = type(current[0]).__name__
            for subtypes in allowed_subtypes:
                if first in subtypes:
                    allowed = subtypes
                    break
            else:
                raise exceptions.JMESPathTypeError(
                    function_name, current[0], first, types)
            for element in current:
                self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
                actual_typename = type(element).__name__
                if actual_typename not in allowed:
                    raise exceptions.JMESPathTypeError(
                        function_name, element, actual_typename, types)

    @signature({'types': ['number']})
    def _func_abs(self, arg):
        return abs(arg)

    @signature({'types': ['array-number']})
    def _func_avg(self, arg):
        if arg:
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION,
                         len(arg))
            return sum(arg) / len(arg)
        else:
            return None

    @signature({'types': [], 'variadic': True})
    def _func_not_null(self, *arguments):
        for argument in arguments:
            if argument is not None:
                return argument

    @signature({'types': []})
    def _func_to_array(self, arg):
        if isinstance(arg, list):
            return arg
        else:
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
            return [arg]

    @signature({'types': []})
    def _func_to_string(self, arg):
        if isinstance(arg, STRING_TYPE):
            return arg
        else:
            return json.dumps(arg, separators=(',', ':'),
                              default=str)

    @signature({'types': []})
    def _func_to_number(self, arg):
        if isinstance(arg, (list, dict, bool)):
            return None
        elif arg is None:
            return None
        elif isinstance(arg, (int, float)):
            return arg
        else:
            try:
                return int(arg)
            except ValueError:
                try:
                    return float(arg)
                except ValueError:
                    return None

    @signature({'types': ['array', 'string']}, {'types': []})
    def _func_contains(self, subject, search):
        budget = self._active_budget
        if isinstance(subject, list):
            # Membership tests on arrays compare against every element
            # until a match is found.
            for element in subject:
                if budget is not None:
                    budget.charge(
                        budget_module.BudgetCategory.ARRAY_ITERATION)
                    budget.charge(
                        budget_module.BudgetCategory.COMPARISON)
                if _equals(element, search):
                    return True
            return False
        return search in subject

    @signature({'types': ['string', 'array', 'object']})
    def _func_length(self, arg):
        return len(arg)

    @signature({'types': ['string']}, {'types': ['string']})
    def _func_ends_with(self, search, suffix):
        return search.endswith(suffix)

    @signature({'types': ['string']}, {'types': ['string']})
    def _func_starts_with(self, search, suffix):
        return search.startswith(suffix)

    @signature({'types': ['array', 'string']})
    def _func_reverse(self, arg):
        if isinstance(arg, STRING_TYPE):
            return arg[::-1]
        else:
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION,
                         len(arg))
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT,
                         len(arg))
            return list(reversed(arg))

    @signature({"types": ['number']})
    def _func_ceil(self, arg):
        return math.ceil(arg)

    @signature({"types": ['number']})
    def _func_floor(self, arg):
        return math.floor(arg)

    @signature({"types": ['string']}, {"types": ['array-string']})
    def _func_join(self, separator, array):
        return separator.join(array)

    @signature({'types': ['expref']}, {'types': ['array']})
    def _func_map(self, expref, arg):
        result = []
        for element in arg:
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION)
            result.append(expref.visit(expref.expression, element))
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT)
        return result

    @signature({"types": ['array-number', 'array-string']})
    def _func_max(self, arg):
        budget = self._active_budget
        if arg:
            if budget is None:
                return max(arg)
            metered = [_MeteredOrderable(value, budget) for value in arg]
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION,
                         len(arg))
            return _metered_best_max(metered).value
        else:
            return None

    @signature({"types": ["object"], "variadic": True})
    def _func_merge(self, *arguments):
        merged = {}
        for arg in arguments:
            self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT,
                         len(arg))
            merged.update(arg)
        return merged

    @signature({"types": ['array-number', 'array-string']})
    @signature({'types': ['array-number', 'array-string']})
    def _func_min(self, arg):
        budget = self._active_budget
        if arg:
            if budget is None:
                return min(arg)
            metered = [_MeteredOrderable(value, budget) for value in arg]
            self._charge(budget_module.BudgetCategory.ARRAY_ITERATION,
                         len(arg))
            return _metered_best_min(metered).value
        else:
            return None


    @signature({"types": ['array-string', 'array-number']})
    @signature({'types': ['array-string', 'array-number']})
    def _func_sort(self, arg):
        budget = self._active_budget
        if budget is None:
            return list(sorted(arg))
        metered = [_MeteredOrderable(value, budget) for value in arg]
        self._charge(budget_module.BudgetCategory.ARRAY_ITERATION, len(arg))
        ordered = list(sorted(metered))
        self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT,
                     len(ordered))
        return [item.value for item in ordered]

    @signature({"types": ['array-number']})
    @signature({'types': ['array-number']})
    def _func_sum(self, arg):
        self._charge(budget_module.BudgetCategory.ARRAY_ITERATION, len(arg))
        return sum(arg)


    @signature({"types": ['object']})
    def _func_keys(self, arg):
        # To be consistent with .values()
        # should we also return the indices of a list?
        self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT,
                     len(arg))
        return list(arg.keys())

    @signature({"types": ['object']})
    def _func_values(self, arg):
        self._charge(budget_module.BudgetCategory.GENERATED_ELEMENT,
                     len(arg))
        return list(arg.values())

    @signature({'types': []})
    def _func_type(self, arg):
        if isinstance(arg, STRING_TYPE):
            return "string"
        elif isinstance(arg, bool):
            return "boolean"
        elif isinstance(arg, list):
            return "array"
        elif isinstance(arg, dict):
            return "object"
        elif isinstance(arg, (float, int)):
            return "number"
        elif arg is None:
            return "null"

    @signature({'types': ['array']}, {'types': ['expref']})
    def _func_sort_by(self, array, expref):
        if not array:
            return array
        budget = self._active_budget
        # sort_by allows for the expref to be either a number or a
        # string, so we have some special logic to handle this.
        # We evaluate the first array element and verify that it's
        # either a string of a number.  We then create a key function
        # that validates that type, which requires that remaining array
        # elements resolve to the same type as the first element.
        if budget is not None:
            budget.charge(budget_module.BudgetCategory.ARRAY_ITERATION)
        required_type = self._convert_to_jmespath_type(
            type(expref.visit(expref.expression, array[0])).__name__)
        if required_type not in ['number', 'string']:
            raise exceptions.JMESPathTypeError(
                'sort_by', array[0], required_type, ['string', 'number'])
        keyfunc = self._create_key_func(expref,
                                        [required_type],
                                        'sort_by')
        if budget is None:
            return list(sorted(array, key=keyfunc))
        # The probe above re-evaluates the first element, then every
        # element (including the first one) is evaluated again for the
        # actual sort: n+1 array iterations in total.
        budget.charge(budget_module.BudgetCategory.ARRAY_ITERATION,
                      len(array))
        keyed = [_MeteredOrderable(keyfunc(element), budget, raw=element)
                 for element in array]
        budget.charge(budget_module.BudgetCategory.GENERATED_ELEMENT,
                      len(array))
        ordered = list(sorted(keyed))
        return [item.raw for item in ordered]

    @signature({'types': ['array']}, {'types': ['expref']})
    def _func_min_by(self, array, expref):
        keyfunc = self._create_key_func(expref,
                                        ['number', 'string'],
                                        'min_by')
        if not array:
            return None
        budget = self._active_budget
        if budget is None:
            return min(array, key=keyfunc)
        return self._budgeted_min_max_by(
            array, keyfunc, budget, pick_max=False)

    @signature({'types': ['array']}, {'types': ['expref']})
    def _func_max_by(self, array, expref):
        keyfunc = self._create_key_func(expref,
                                        ['number', 'string'],
                                        'max_by')
        if not array:
            return None
        budget = self._active_budget
        if budget is None:
            return max(array, key=keyfunc)
        return self._budgeted_min_max_by(
            array, keyfunc, budget, pick_max=True)

    def _budgeted_min_max_by(self, array, keyfunc, budget, pick_max):
        # Every element key is evaluated once (n array iterations) and a
        # non-empty result performs n-1 key comparisons.  An empty array
        # is handled by the callers and consumes no work here.
        budget.charge(budget_module.BudgetCategory.ARRAY_ITERATION,
                      len(array))
        keyed = [_MeteredOrderable(keyfunc(element), budget, raw=element)
                 for element in array]
        if pick_max:
            return _metered_best_max(keyed).raw
        return _metered_best_min(keyed).raw

    def _create_key_func(self, expref, allowed_types, function_name):
        def keyfunc(x):
            result = expref.visit(expref.expression, x)
            actual_typename = type(result).__name__
            jmespath_type = self._convert_to_jmespath_type(actual_typename)
            # allowed_types is in term of jmespath types, not python types.
            if jmespath_type not in allowed_types:
                raise exceptions.JMESPathTypeError(
                    function_name, result, jmespath_type, allowed_types)
            return result
        return keyfunc

    def _convert_to_jmespath_type(self, pyobject):
        return TYPES_MAP.get(pyobject, 'unknown')
