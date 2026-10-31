import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from rewards import ToolReward

model = "Qwen/Qwen3.5-2B"
revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
model_eos = AutoConfig.from_pretrained(model, revision=revision, local_files_only=True).get_text_config().eos_token_id
reward = ToolReward(tokenizer, "exact", eos_token_ids=model_eos)
rows = []
for split in ("train", "validation"):
    rows.extend(json.loads(line) | {"split": split} for line in Path(f"data/{split}.jsonl").read_text().splitlines())
sample_ids = {row["id"] for row in [x for x in rows if x["category"] == "call"][:100] + [x for x in rows if x["category"] == "no_call"][:20]}
failures = []
stats = Counter()
sample_failures = []
numeric_normalizations = 0
for row in rows:
    calls = json.loads(row["expected"])
    tools = json.loads(row["tools_json"])
    stats[row["split"] + ":" + row["category"]] += 1
    stats["empty_tools"] += not tools
    stats["parallel_rows"] += len(calls) > 1
    def inspect(value):
        if isinstance(value, str):
            stats["string_arguments"] += 1
        elif isinstance(value, bool):
            stats["boolean_arguments"] += 1
        elif isinstance(value, (list, dict)):
            stats["nested_arguments"] += 1
            for item in value.values() if isinstance(value, dict) else value:
                inspect(item)
    for call in calls:
        for value in call["arguments"].values():
            inspect(value)
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call} for call in calls]} if calls else {"role": "assistant", "content": "Please provide the missing information."}
    messages = [{"role": "user", "content": "Check reference formatting."}]
    prefix = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages + [assistant], tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prefix):
        failures.append({"id": row["id"], "error": "template_prefix_mismatch"})
        continue
    text = full[len(prefix):]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    score = reward([text], [row["expected"]], [row["tools_json"]])[0]
    raw_score = reward([""], [row["expected"]], [row["tools_json"]], completion_ids=[ids])[0]
    request = reward.request_type.model_validate({"model": model, "messages": messages, **({"tools": tools} if tools else {})})
    reward.parser.tools = request.tools or []
    parsed = reward.parser.extract_tool_calls(text, request)
    actual = [{"name": call.function.name, "arguments": json.loads(call.function.arguments)} for call in parsed.tool_calls]
    key = lambda value: sorted(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in value)
    numeric_normalizations += key(actual) != key(calls) and score == 1
    if score != 1 or raw_score != 1:
        failure = {"id": row["id"], "split": row["split"], "category": row["category"], "score": score, "raw_score": raw_score, "expected": calls, "parsed": actual, "response": text}
        failures.append(failure)
        if row["id"] in sample_ids:
            sample_failures.append(failure)
result = {
    "model": model, "revision": revision, "cuda_visible_devices": "", "gpu_initialized": torch.cuda.is_initialized(),
    "rows": len(rows), "sample_rows": len(sample_ids), "sample_failures": len(sample_failures),
    "failures": len(failures), "statistics": dict(stats), "numeric_parser_normalizations": numeric_normalizations,
    "tokenizer_eos": tokenizer.eos_token_id, "model_eos": model_eos,
    "eos_decoded": {str(token): tokenizer.decode([token], skip_special_tokens=False) for token in reward.eos_token_ids},
    "no_call_text": "Synthetic nonempty clarification; scoring checks no-call structure, not prose semantics.",
    "data_sha256": {split: hashlib.sha256(Path(f"data/{split}.jsonl").read_bytes()).hexdigest() for split in ("train", "validation")},
    "reward_sha256": hashlib.sha256(Path("rewards.py").read_bytes()).hexdigest(),
    "versions": {package: version(package) for package in ("vllm", "transformers", "trl")},
    "failure_details": failures,
}
Path("outputs/reward_integration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "failure_details"}, ensure_ascii=False))
if failures:
    print(json.dumps(failures[:5], ensure_ascii=False))
    raise SystemExit(1)
