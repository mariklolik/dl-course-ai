import unittest
from types import SimpleNamespace

from rewards import ToolReward, score_calls


def response(name="weather", value="Paris"):
    return (
        f"<tool_call><function={name}><parameter=city>{value}</parameter>"
        "</function></tool_call>"
    )


def reward_adapter():
    reward = object.__new__(ToolReward)
    reward.mode = "exact"
    reward.request_type = SimpleNamespace(model_validate=lambda value: SimpleNamespace(tools=value.get("tools")))
    parsed = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name="weather", arguments='{"city":"Paris"}'))])
    reward.parser = SimpleNamespace(extract_tool_calls=lambda text, request: parsed)
    reward.tokenizer = SimpleNamespace(
        eos_token_id=2,
        decode=lambda ids, skip_special_tokens: "" if skip_special_tokens else response() + "<|im_end|>",
    )
    reward.eos_token_ids = {2}
    return reward


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.call = {"name": "weather", "arguments": {"city": "Paris"}}

    def test_parallel_order_does_not_change_exact_reward(self):
        other = {"name": "weather", "arguments": {"city": "London"}}
        text = response() + response(value="London")
        self.assertEqual(score_calls(text, [self.call, other], [other, self.call], "exact"), 1)
        self.assertAlmostEqual(score_calls(text, [self.call, other], [other, self.call]), 1)

    def test_duplicate_calls_cannot_disappear(self):
        self.assertEqual(score_calls(response(), [self.call], [self.call, self.call], "exact"), 0)
        self.assertEqual(score_calls(response() * 2, [self.call] * 2, [self.call], "exact"), 0)
        self.assertAlmostEqual(score_calls(response(), [self.call], [self.call] * 2), 0.15 + 0.55 * 2 / 3)

    def test_argument_types_remain_distinct_at_every_depth(self):
        for actual, expected in [(True, 1), (1, "1"), (None, "null")]:
            with self.subTest(actual=actual, expected=expected):
                calls = [{"name": "weather", "arguments": {"x": [{"a": actual}]}}]
                target = [{"name": "weather", "arguments": {"x": [{"a": expected}]}}]
                self.assertEqual(score_calls(response(), calls, target, "exact"), 0)
                self.assertAlmostEqual(score_calls(response(), calls, target), 0.4)

    def test_integral_json_numbers_match_official_parser_normalization(self):
        calls = [{"name": "weather", "arguments": {"x": [{"a": 10}]}}]
        target = [{"name": "weather", "arguments": {"x": [{"a": 10.0}]}}]
        self.assertEqual(score_calls(response(), calls, target, "exact"), 1)
        self.assertEqual(score_calls(response(), calls, target), 1)

    def test_partial_arguments_reward_only_matching_values(self):
        calls = [{"name": "weather", "arguments": {"city": "Paris", "days": 2}}]
        target = [{"name": "weather", "arguments": {"city": "Paris", "days": 3}}]
        self.assertAlmostEqual(score_calls(response(), calls, target), 0.55)

    def test_argument_object_order_is_irrelevant(self):
        calls = [{"name": "weather", "arguments": {"city": "Paris", "nested": {"a": 1, "b": [2]}}}]
        target = [{"name": "weather", "arguments": {"nested": {"b": [2], "a": 1}, "city": "Paris"}}]
        self.assertEqual(score_calls(response(), calls, target, "exact"), 1)

    def test_argument_array_order_is_significant(self):
        calls = [{"name": "weather", "arguments": {"x": [1, 2]}}]
        target = [{"name": "weather", "arguments": {"x": [2, 1]}}]
        self.assertEqual(score_calls(response(), calls, target, "exact"), 0)

    def test_empty_containers_are_not_omitted_arguments(self):
        for value in ({}, []):
            calls = [{"name": "weather", "arguments": {"x": value}}]
            target = [{"name": "weather", "arguments": {}}]
            self.assertEqual(score_calls(response(), calls, target, "exact"), 0)

    def test_irrelevant_schema_correct_call_gets_no_reward(self):
        calls = [{"name": "other", "arguments": {"city": "Paris"}}]
        self.assertEqual(score_calls(response("other"), calls, [self.call]), 0)

    def test_missing_required_call_gets_no_reward(self):
        self.assertEqual(score_calls("I cannot help.", [], [self.call]), 0)

    def test_no_call_target_requires_nonempty_text_and_no_spurious_calls(self):
        for mode in ("exact", "shaped"):
            self.assertEqual(score_calls("Please clarify.", [], [], mode), 1)
            self.assertEqual(score_calls("", [], [], mode), 0)
            self.assertEqual(score_calls("   ", [], [], mode), 0)
            self.assertEqual(score_calls(response(), [self.call], [], mode), 0)

    def test_parser_failure_and_unparsed_markup_get_no_reward(self):
        for text, calls in [
            (response(), None),
            (response(), []),
            ("<tool_call>broken", None),
            ("<tool_call>broken</tool_call>", []),
            ("<function=weather></function>", []),
            ("<parameter=city>Paris</parameter>", []),
            (response() + "<tool_call", [self.call]),
            (response().replace("</function>", ""), [self.call]),
            (response() + "</tool_call>", [self.call]),
            (response() + "<parameterx>", [self.call]),
            (response() + "<parameter=city>London</parameter>", [self.call]),
            (response() + "<function=other></function>", [self.call]),
        ]:
            with self.subTest(text=text):
                self.assertEqual(score_calls(text, calls, [self.call]), 0)
                self.assertEqual(score_calls(text, calls, []), 0)

    def test_exact_reward_cannot_credit_cross_call_argument_swaps(self):
        calls = [
            {"name": "weather", "arguments": {"city": "Paris", "days": 2}},
            {"name": "weather", "arguments": {"city": "London", "days": 3}},
        ]
        target = [
            {"name": "weather", "arguments": {"city": "Paris", "days": 3}},
            {"name": "weather", "arguments": {"city": "London", "days": 2}},
        ]
        self.assertEqual(score_calls(response() * 2, calls, target, "exact"), 0)
        self.assertLess(score_calls(response() * 2, calls, target), 1)

    def test_adapter_rejects_explicit_truncation_even_with_balanced_xml(self):
        reward = reward_adapter()
        scores = reward([response()], ['[{"name":"weather","arguments":{"city":"Paris"}}]'], ['[]'], finish_reason=["length"])
        self.assertEqual(scores, [0])

    def test_adapter_decodes_raw_ids_with_special_xml_tokens(self):
        reward = reward_adapter()
        scores = reward([""], ['[{"name":"weather","arguments":{"city":"Paris"}}]'], ['[]'], completion_ids=[[1, 2]])
        self.assertEqual(scores, [1])

    def test_adapter_rejects_ids_without_terminal_eos(self):
        reward = reward_adapter()
        scores = reward([response()], ['[{"name":"weather","arguments":{"city":"Paris"}}]'], ['[]'], completion_ids=[[1]])
        self.assertEqual(scores, [0])

    def test_adapter_accepts_additional_model_eos(self):
        reward = reward_adapter()
        reward.eos_token_ids = {2, 3}
        scores = reward([response()], ['[{"name":"weather","arguments":{"city":"Paris"}}]'], ['[]'], completion_ids=[[1, 3]])
        self.assertEqual(scores, [1])

    def test_adapter_omits_empty_tool_lists_from_request(self):
        reward = reward_adapter()

        def validate(value):
            if value.get("tools") == []:
                raise ValueError("Empty tools are forbidden")
            return SimpleNamespace(tools=value.get("tools"))

        reward.request_type = SimpleNamespace(model_validate=validate)
        reward.parser = SimpleNamespace(extract_tool_calls=lambda text, request: SimpleNamespace(tool_calls=[]))
        self.assertEqual(reward(["Please clarify."], ["[]"], ["[]"]), [1])

    def test_control_tokens_alone_are_not_a_no_call_answer(self):
        self.assertEqual(score_calls("<|im_end|>", [], []), 0)
        self.assertEqual(score_calls("<think>hidden</think><|im_end|>", [], []), 0)


if __name__ == "__main__":
    unittest.main()
