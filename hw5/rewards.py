import json
import re
from collections import Counter


def normalize_numbers(value):
    if isinstance(value, dict):
        return {key: normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    return int(value) if isinstance(value, float) and value.is_integer() else value


def canonical(value):
    return json.dumps(normalize_numbers(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def argument_leaves(value, path=()):
    if isinstance(value, dict) and value:
        return [leaf for key, item in value.items() for leaf in argument_leaves(item, (*path, key))]
    if isinstance(value, list) and value:
        return [leaf for index, item in enumerate(value) for leaf in argument_leaves(item, (*path, index))]
    return [(path, canonical(value))]


def multiset_f1(actual, expected):
    total = sum(actual.values()) + sum(expected.values())
    return 2 * sum((actual & expected).values()) / total if total else 1.0


def complete_response(response, calls):
    visible = re.sub(r"<\|[^>]*\|>", "", response)
    visible = re.sub(r"<think>.*?</think>", "", visible, flags=re.DOTALL)
    if calls is None or not visible.strip():
        return False
    pairs = [("<tool_call>", "</tool_call>"), ("<function=", "</function>"), ("<parameter=", "</parameter>")]
    if any(response.count(start) != response.count(end) for start, end in pairs):
        return False
    if response.count("<tool_call>") != len(calls):
        return False
    pattern = r"</?tool_call>|<function=[^<>]+>|</function>|<parameter=[^<>]+>|</parameter>"
    remaining = re.sub(pattern, "", response)
    if re.search(r"</?(?:tool|function|parameter)", remaining):
        return False
    stack = []
    parents = {"tool_call": None, "function": "tool_call", "parameter": "function"}
    for tag in re.findall(pattern, response):
        name = tag.lstrip("</").rstrip(">").split("=", 1)[0]
        if tag.startswith("</"):
            if not stack or stack.pop() != name:
                return False
        else:
            if parents[name] != (stack[-1] if stack else None):
                return False
            stack.append(name)
    if stack:
        return False
    return bool(calls) or not any(tag in response for tag in ("<tool", "<function", "<parameter"))


def score_calls(response: str, calls: list[dict] | None, expected: list[dict], mode: str = "shaped") -> float:
    if mode not in {"exact", "shaped"}:
        raise ValueError(f"Unknown reward mode: {mode}")
    if not complete_response(response, calls):
        return 0.0
    if not expected:
        return float(not calls)
    if not calls:
        return 0.0
    actual_names = Counter(call["name"] for call in calls)
    expected_names = Counter(call["name"] for call in expected)
    if not actual_names & expected_names:
        return 0.0
    exact = Counter((call["name"], canonical(call["arguments"])) for call in calls) == Counter(
        (call["name"], canonical(call["arguments"])) for call in expected
    )
    if mode == "exact":
        return float(exact)
    actual_arguments = Counter((call["name"], path, value) for call in calls for path, value in argument_leaves(call["arguments"]))
    expected_arguments = Counter((call["name"], path, value) for call in expected for path, value in argument_leaves(call["arguments"]))
    return 0.15 + 0.25 * multiset_f1(actual_names, expected_names) + 0.30 * multiset_f1(actual_arguments, expected_arguments) + 0.30 * exact


class ToolReward:
    def __init__(self, tokenizer, mode="shaped", eos_token_ids=None):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

        if mode not in {"exact", "shaped"}:
            raise ValueError(f"Unknown reward mode: {mode}")
        self.tokenizer = tokenizer
        self.mode = mode
        self.parser = Qwen3CoderToolParser(tokenizer)
        self.request_type = ChatCompletionRequest
        eos = tokenizer.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.eos_token_ids.update([eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids or [])
        self.__name__ = f"tool_{mode}"

    def __call__(self, completions, expected, tools_json, completion_ids=None, **kwargs):
        rewards = []
        finish_reasons = kwargs.get("finish_reason", kwargs.get("finish_reasons"))
        for index, (completion, target, tools) in enumerate(zip(completions, expected, tools_json, strict=True)):
            text = completion if isinstance(completion, str) else completion[-1].get("content", "")
            if completion_ids is not None:
                ids = completion_ids[index]
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                if not any(token in self.eos_token_ids for token in ids):
                    rewards.append(0.0)
                    continue
            if finish_reasons and finish_reasons[index] == "length":
                rewards.append(0.0)
                continue
            try:
                schemas = json.loads(tools)
                request = self.request_type.model_validate({
                    "model": "Qwen/Qwen3.5-2B", "messages": [], **({"tools": schemas} if schemas else {}),
                })
                self.parser.tools = request.tools or []
                parsed = self.parser.extract_tool_calls(text, request)
                calls = [{"name": call.function.name, "arguments": json.loads(call.function.arguments)} for call in parsed.tool_calls]
                rewards.append(score_calls(text, calls, json.loads(target), self.mode))
            except (ValueError, TypeError, KeyError, AttributeError):
                rewards.append(0.0)
        return rewards
