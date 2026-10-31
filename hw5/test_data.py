import json
import unittest

import prepare_data as data


def source(turns, names=('Market Trends API',)):
    tools = [{'name': name, 'description': 'lookup', 'parameters': {'type': 'dict', 'properties': {'q': {'type': 'string'}}, 'required': ['q']}, 'required': None} for name in names]
    return {'system': data.MARKER + '\n' + json.dumps(tools) + '. legacy format instructions', 'conversations': [{'from': role, 'value': text} for role, text in turns]}


class DataTests(unittest.TestCase):
    def test_extracts_nested_schema_and_normalizes_function_name(self):
        tools, names = data.extract_tools(source([])['system'])
        self.assertEqual(tools[0]['function']['name'], 'Market_Trends_API')
        self.assertEqual(tools[0]['function']['parameters']['type'], 'object')
        self.assertEqual(names, {'Market Trends API': 'Market_Trends_API'})

    def test_ast_parsing_preserves_literals_and_parallel_duplicates(self):
        names = {'Market Trends API': 'Market_Trends_API'}
        calls = data.parse_calls('[Market Trends API(q="a), b(", extra={"x": [-2, True, None]}), Market Trends API(q="again")]', names)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['arguments'], {'q': 'a), b(', 'extra': {'x': [-2, True, None]}})

    def test_rejects_executable_or_positional_call_arguments(self):
        names = {'tool': 'tool'}
        for value in ['[tool(q=__import__("os"))]', '[tool("x")]', '[unknown(q="x")]', '[tool(q=1, q=2)]']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.parse_calls(value, names)

    def test_name_collisions_remain_distinct(self):
        tools, names = data.extract_tools(source([], ('a-b', 'a_b_2', 'a b'))['system'])
        self.assertEqual(len(set(names.values())), 3)
        self.assertEqual(len(tools), 3)

    def test_only_user_decisions_become_targets_and_history_is_prefix(self):
        row = source([('user', 'first'), ('assistant', '[Market Trends API(q="x")]'), ('tool', '[{"name":"Market Trends API","results":{"x":1}}]'), ('assistant', 'acknowledged'), ('user', 'second'), ('assistant', 'missing information')])
        examples, audit = data.extract_examples([row])
        self.assertEqual([e['category'] for e in examples], ['call', 'no_call'])
        self.assertEqual(examples[0]['messages'][-1]['content'], 'first')
        self.assertEqual(examples[1]['messages'][-1]['content'], 'second')
        self.assertNotIn('missing information', str(examples[1]['messages']))
        self.assertEqual(audit['excluded']['post_tool_assistant'], 1)

    def test_duplicate_dialogues_cannot_cross_split(self):
        rows = [source([('user', str(i)), ('assistant', 'no tool needed')]) for i in range(20)]
        examples, audit = data.extract_examples(rows + [rows[0]])
        train, valid = data.split_dialogues(examples, seed=42)
        self.assertFalse({r['dialogue_id'] for r in train} & {r['dialogue_id'] for r in valid})
        self.assertEqual(audit['duplicate_dialogues'], 1)
        self.assertEqual(len(examples), 20)

    def test_preserves_source_date_context(self):
        row = source([('user', 'tomorrow'), ('assistant', 'need location')])
        row['system'] = 'also point it out.\nToday is 2024-01-06, Saturday..' + row['system']
        examples, audit = data.extract_examples([row])
        self.assertIn('2024-01-06', examples[0]['messages'][0]['content'])

    def test_unsupported_schema_is_explicit_and_empty_tools_are_valid(self):
        row = source([('user', 'hello'), ('assistant', 'hello')], ())
        examples, audit = data.extract_examples([row, {'system': '<table>', 'conversations': []}])
        self.assertEqual(examples[0]['expected'], '[]')
        self.assertEqual(audit['excluded']['unsupported_tool_schema'], 1)

    def test_balanced_selection_is_deterministic_and_uses_requested_fraction(self):
        pool = [{'category': 'no_call' if i < 10 else 'call', 'id': i} for i in range(40)]
        selected = data.select_balanced(pool, 20)
        self.assertEqual(sum(r['category'] == 'no_call' for r in selected), 5)
        self.assertEqual(selected, data.select_balanced(pool, 20))


if __name__ == '__main__':
    unittest.main()
