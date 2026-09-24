import threading
import unittest

import jmespath
from jmespath import budget
from jmespath import exceptions
from jmespath import functions


def make_options(**limits):
    return jmespath.Options(budget=limits)


def consumed_after(expression, data, **limits):
    """Evaluate and return the per-category consumption of the call."""
    if not limits:
        limits = {'total': 10 ** 12}
    jmespath.search(expression, data, make_options(**limits))
    context = budget.current_context()
    assert context is not None
    return context.consumed


class TestExactCountingSemantics(unittest.TestCase):
    def test_projection_counts_elements_and_non_null_outputs(self):
        # Null projection results are filtered out: every input element
        # is iterated, but only non-null results become output elements.
        data = [{'a': 1}, {'b': 2}, {'a': None}, {'a': 4}]
        self.assertEqual(jmespath.search('[*].a', data), [1, 4])
        consumed = consumed_after('[*].a', data)
        self.assertEqual(consumed['elements'], 4)
        self.assertEqual(consumed['output_elements'], 2)
        # projection + identity + one field visit per element.
        self.assertEqual(consumed['ast_nodes'], 6)
        self.assertEqual(consumed['comparisons'], 0)
        self.assertEqual(consumed['function_calls'], 0)

    def test_or_expression_short_circuits(self):
        # The right hand side of || is only evaluated when the left
        # hand side is falsy.
        consumed = consumed_after('a || b', {'a': 1, 'b': 2})
        self.assertEqual(consumed['ast_nodes'], 2)  # or + field a
        consumed = consumed_after('a || b', {'a': [], 'b': 5})
        self.assertEqual(consumed['ast_nodes'], 3)  # or + field a + field b

    def test_pipe_materializes_left_side(self):
        # The left side of a pipe is fully materialized (counted as
        # output elements) before the right side is evaluated.
        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        self.assertEqual(jmespath.search('[*].a | [0]', data), 1)
        consumed = consumed_after('[*].a | [0]', data)
        self.assertEqual(consumed['elements'], 3)
        self.assertEqual(consumed['output_elements'], 3)

    def test_sort_by_is_stable_and_counts_key_evaluations(self):
        data = [{'k': 1, 'v': 'a'}, {'k': 1, 'v': 'b'}, {'k': 0, 'v': 'c'}]
        result = jmespath.search('sort_by(@, &k)', data)
        # Equal keys keep their original relative order (stable sort).
        self.assertEqual([item['v'] for item in result], ['c', 'a', 'b'])
        consumed = consumed_after('sort_by(@, &k)', data)
        self.assertEqual(consumed['function_calls'], 1)
        # One key evaluation (comparison) per element.
        self.assertEqual(consumed['comparisons'], 3)
        self.assertEqual(consumed['output_elements'], 3)

    def test_max_by_on_empty_array(self):
        # max_by of an empty array is null and performs no key
        # evaluations at all.
        self.assertIsNone(jmespath.search('max_by(`[]`, &a)', {}))
        consumed = consumed_after('max_by(`[]`, &a)', {})
        self.assertEqual(consumed['function_calls'], 1)
        self.assertEqual(consumed['comparisons'], 0)
        self.assertEqual(consumed['elements'], 0)
        self.assertEqual(consumed['output_elements'], 0)

    def test_flatten_counts_input_and_output_elements(self):
        data = [[1, 2], [3], 4]
        self.assertEqual(jmespath.search('[]', data), [1, 2, 3, 4])
        consumed = consumed_after('[]', data)
        # "[]" is a flatten followed by an identity projection: the
        # flatten iterates 3 elements and emits 4, then the projection
        # iterates and emits the 4 flattened elements.
        self.assertEqual(consumed['elements'], 3 + 4)
        self.assertEqual(consumed['output_elements'], 4 + 4)

    def test_filter_projection_counts_comparisons(self):
        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        result = jmespath.search('[?a > `1`].a', data)
        self.assertEqual(result, [2, 3])
        consumed = consumed_after('[?a > `1`].a', data)
        self.assertEqual(consumed['elements'], 3)
        self.assertEqual(consumed['comparisons'], 3)
        self.assertEqual(consumed['output_elements'], 2)


class TestBudgetLimits(unittest.TestCase):
    def test_no_budget_by_default(self):
        self.assertIsNone(jmespath.Options().budget)
        jmespath.search('[*].a', [{'a': 1}])
        self.assertIsNone(budget.current_context())

    def test_exact_limit_is_allowed(self):
        # Endpoint semantics: consuming exactly the limit succeeds;
        # only strictly exceeding it fails.
        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        result = jmespath.search('[*].a', data, make_options(elements=3))
        self.assertEqual(result, [1, 2, 3])

    def test_exceeding_category_limit_raises(self):
        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('[*].a', data, make_options(elements=2))
        error = cm.exception
        self.assertEqual(error.category, 'elements')
        self.assertEqual(error.consumed, 3)
        self.assertEqual(error.limit, 2)

    def test_exceeding_total_limit_raises(self):
        data = [{'a': 1}, {'a': 2}, {'a': 3}]
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('[*].a', data, make_options(total=5))
        self.assertEqual(cm.exception.category, 'total')

    def test_function_call_limit(self):
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('length(a)', {'a': [1]},
                            make_options(function_calls=0))
        self.assertEqual(cm.exception.category, 'function_calls')

    def test_comparison_limit(self):
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('[?a > `1`]', [{'a': 2}],
                            make_options(comparisons=0))
        self.assertEqual(cm.exception.category, 'comparisons')

    def test_output_element_limit(self):
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('[*].a', [{'a': 1}, {'a': 2}],
                            make_options(output_elements=1))
        self.assertEqual(cm.exception.category, 'output_elements')

    def test_ast_node_limit(self):
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('a.b.c', {'a': {'b': {'c': 1}}},
                            make_options(ast_nodes=2))
        self.assertEqual(cm.exception.category, 'ast_nodes')

    def test_budget_accepts_budget_limits_instance(self):
        limits = jmespath.BudgetLimits(total=100, elements=10)
        options = jmespath.Options(budget=limits)
        self.assertIs(options.budget, limits)
        self.assertEqual(
            jmespath.search('[*]', [1, 2], options), [1, 2])

    def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            jmespath.BudgetLimits(total=-1)
        with self.assertRaises(ValueError):
            jmespath.BudgetLimits(elements='lots')
        with self.assertRaises(ValueError):
            jmespath.Options(budget=42)


class TestDiagnostics(unittest.TestCase):
    def test_error_contains_position_and_consumption(self):
        data = {'items': [{'v': 1}, {'v': 2}, {'v': 3}]}
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('items[*].v', data, make_options(elements=2))
        error = cm.exception
        message = str(error)
        self.assertIn('elements', message)
        self.assertIn('consumed 3', message)
        self.assertIn('limit is 2', message)
        # Expression position information: the AST path and the
        # expression itself are included.
        self.assertIn('items[*].v', message)
        self.assertIn('projection', error.diagnostics['ast_path'])
        # Current data path at the point of failure.
        self.assertEqual(error.diagnostics['data_path'], '$[2]')
        self.assertEqual(error.diagnostics['consumed']['elements'], 3)

    def test_error_does_not_serialize_user_data(self):
        secret = 'S3CR3T-USER-VALUE'
        data = {'items': [{'v': secret}] * 3}
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('items[*].v', data, make_options(elements=1))
        self.assertNotIn(secret, str(cm.exception))


class TestCancellation(unittest.TestCase):
    def test_cancel_from_custom_function(self):
        class CancellingFunctions(functions.Functions):
            @functions.signature({'types': []})
            def _func_stop(self, arg):
                budget.current_context().cancel('user asked to stop')
                return arg

        options = jmespath.Options(
            custom_functions=CancellingFunctions(),
            budget={'total': 10 ** 6})
        with self.assertRaises(exceptions.EvaluationCancelledError) as cm:
            jmespath.search('stop(a) | b', {'a': 1, 'b': 2}, options)
        self.assertEqual(cm.exception.reason, 'user asked to stop')
        self.assertIn('ast_path', cm.exception.diagnostics)

    def test_should_cancel_callback(self):
        calls = []

        def should_cancel():
            calls.append(1)
            return len(calls) > 3

        options = make_options(total=10 ** 6, should_cancel=should_cancel)
        with self.assertRaises(exceptions.EvaluationCancelledError):
            jmespath.search('[*].a', [{'a': 1}] * 10, options)

    def test_cancellation_is_distinct_from_other_errors(self):
        self.assertFalse(issubclass(exceptions.EvaluationCancelledError,
                                    exceptions.BudgetExceededError))
        self.assertFalse(issubclass(exceptions.BudgetExceededError,
                                    exceptions.EvaluationCancelledError))
        self.assertFalse(issubclass(exceptions.EvaluationCancelledError,
                                    exceptions.JMESPathTypeError))
        self.assertFalse(issubclass(exceptions.EvaluationCancelledError,
                                    exceptions.ArityError))
        self.assertFalse(issubclass(exceptions.BudgetExceededError,
                                    exceptions.JMESPathTypeError))
        # Type and arity errors still behave as before under a budget.
        options = make_options(total=10 ** 6)
        with self.assertRaises(exceptions.JMESPathTypeError):
            jmespath.search('abs(`"str"`)', {}, options)
        with self.assertRaises(exceptions.ArityError):
            jmespath.search('abs(`1`, `2`)', {}, options)


class RecordingFunctions(functions.Functions):
    """Custom function that debits its own work to the budget."""
    @functions.signature({'types': ['array']})
    def _func_scan(self, array):
        context = budget.current_context()
        for _ in array:
            context.consume('elements')
        return len(array)


class NestedSearchFunctions(functions.Functions):
    """Custom function that recursively evaluates via the context."""
    @functions.signature({'types': []})
    def _func_nested(self, value):
        return budget.current_context().search('a.b', value)


class TestCustomFunctionIntegration(unittest.TestCase):
    def test_custom_function_debits_shared_budget(self):
        options = jmespath.Options(
            custom_functions=RecordingFunctions(),
            budget={'elements': 3})
        # 3 elements: exactly at the limit, allowed.
        result = jmespath.search('scan(@)', [1, 2, 3], options)
        self.assertEqual(result, 3)
        with self.assertRaises(exceptions.BudgetExceededError):
            jmespath.search('scan(@)', [1, 2, 3, 4], options)

    def test_nested_search_shares_parent_budget(self):
        options = jmespath.Options(
            custom_functions=NestedSearchFunctions(),
            budget={'ast_nodes': 4})
        data = {'a': {'b': 1}}
        # nested(@) evaluates a.b under the same budget:
        # function_expression + current + subexpression + 2 fields = 5.
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('nested(@)', data, options)
        self.assertEqual(cm.exception.category, 'ast_nodes')
        # With a sufficient budget the nested evaluation succeeds.
        options = jmespath.Options(
            custom_functions=NestedSearchFunctions(),
            budget={'ast_nodes': 5})
        self.assertEqual(jmespath.search('nested(@)', data, options), 1)

    def test_no_context_outside_budgeted_evaluation(self):
        # A non-budgeted search clears the current context, so
        # current_context() is None afterwards.
        jmespath.search('a', {'a': 1})
        self.assertIsNone(budget.current_context())


class TestPerCallBudgetState(unittest.TestCase):
    def test_budget_state_is_fresh_per_search_call(self):
        parsed = jmespath.compile('[*].a')
        options = make_options(elements=100)
        parsed.search([{'a': 1}] * 10, options=options)
        first = budget.current_context().consumed['elements']
        parsed.search([{'a': 1}] * 5, options=options)
        second = budget.current_context().consumed['elements']
        self.assertEqual(first, 10)
        # The second call does not accumulate on top of the first.
        self.assertEqual(second, 5)

    def test_cached_ast_usable_concurrently_with_different_budgets(self):
        # The parser cache shares ASTs between threads; budget state
        # must remain local to each search call.
        parsed = jmespath.compile('items[*].v')
        data = {'items': [{'v': i} for i in range(50)]}
        results = {}

        def run(name, element_limit):
            options = make_options(elements=element_limit)
            try:
                results[name] = parsed.search(data, options=options)
            except exceptions.BudgetExceededError as e:
                results[name] = e

        threads = [
            threading.Thread(target=run, args=('tight', 10)),
            threading.Thread(target=run, args=('loose', 1000)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertIsInstance(results['tight'],
                              exceptions.BudgetExceededError)
        self.assertEqual(results['loose'], list(range(50)))


class TestDeterministicGenerators(unittest.TestCase):
    def test_deep_expression(self):
        # Deterministically generated deep field chain.
        depth = 50
        expression = '.'.join(['f'] * depth)
        data = current = {}
        for _ in range(depth):
            current['f'] = {}
            current = current['f']
        current['leaf'] = True
        # One visit per field node plus one for the subexpression node.
        consumed = consumed_after(expression, data)
        self.assertEqual(consumed['ast_nodes'], depth + 1)
        with self.assertRaises(exceptions.BudgetExceededError):
            jmespath.search(expression, data,
                            make_options(ast_nodes=depth))
        # Exactly at the limit: allowed.
        jmespath.search(expression, data, make_options(ast_nodes=depth + 1))

    def test_wide_projection(self):
        # Deterministically generated wide data array.
        width = 1000
        data = {'items': [{'v': i} for i in range(width)]}
        consumed = consumed_after('items[*].v', data)
        self.assertEqual(consumed['elements'], width)
        self.assertEqual(consumed['output_elements'], width)
        with self.assertRaises(exceptions.BudgetExceededError) as cm:
            jmespath.search('items[*].v', data,
                            make_options(elements=width - 1))
        self.assertEqual(cm.exception.consumed, width)
        jmespath.search('items[*].v', data, make_options(elements=width))

    def test_wide_multiselect(self):
        width = 100
        expression = '[%s]' % ', '.join(['a'] * width)
        consumed = consumed_after(expression, {'a': 1})
        self.assertEqual(consumed['output_elements'], width)
        with self.assertRaises(exceptions.BudgetExceededError):
            jmespath.search(expression, {'a': 1},
                            make_options(output_elements=width - 1))


class TestDefaultBehaviorUnchanged(unittest.TestCase):
    def test_results_identical_with_and_without_budget(self):
        expressions = [
            'foo.bar[*].baz',
            'foo[?bar > `1`].bar',
            'sort_by(foo, &bar)[*]',
            'foo || bar',
            'foo[*].[a, b]',
            'max_by(foo, &bar)',
            'length(foo[*])',
        ]
        data = {
            'foo': [{'bar': 3, 'baz': 1, 'a': 1, 'b': 2},
                    {'bar': 1, 'baz': 2, 'a': 3, 'b': 4}],
            'bar': 'fallback',
        }
        options = make_options(total=10 ** 9)
        for expression in expressions:
            self.assertEqual(
                jmespath.search(expression, data),
                jmespath.search(expression, data, options),
                expression)


if __name__ == '__main__':
    unittest.main()
