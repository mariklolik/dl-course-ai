import json
from pathlib import Path
from types import SimpleNamespace


def load_functions():
    path = Path(__file__).with_name("sft_function_calling_hw.ipynb")
    notebook = json.loads((path if path.exists() else path.parent / ".work/template.ipynb").read_text())
    namespace = {"torch": SimpleNamespace(inference_mode=lambda: lambda function: function)}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or not any(name in source for name in ("def generate(", "def parse_tool_calls(", "def run_agent(")):
            continue
        exec("import re\nimport json\nfrom smart_home import *\n" + source, namespace)
    return namespace


def test_generation_contract():
    functions = load_functions()
    calls = {}

    class Tokens:
        shape = (1, 3)

        def to(self, device):
            calls["device"] = device
            return self

    class Sequence:
        def __getitem__(self, item):
            assert item == (0, slice(3, None))
            return [11, 12]

    class Tokenizer:
        eos_token_id = 151645

        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return {"input_ids": Tokens(), "attention_mask": Tokens()}

        def decode(self, tokens, **kwargs):
            assert tokens == [11, 12] and kwargs == {"skip_special_tokens": True}
            return "answer"

    class Model:
        device = "cpu"

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generation"] = kwargs
            return Sequence()

    messages = [{"role": "user", "content": "hello"}]
    assert functions["generate"](Model(), Tokenizer(), messages, [], 13) == "answer"
    assert calls["template"][1] == {
        "tools": [], "add_generation_prompt": True, "tokenize": True,
        "return_dict": True, "return_tensors": "pt",
    }
    assert calls["generation"]["do_sample"] is False
    assert calls["generation"]["eos_token_id"] == 151645
    assert calls["generation"]["max_new_tokens"] == 13
    assert calls["eval"]


def test_agent_executes_all_calls_and_keeps_errors():
    functions = load_functions()
    outputs = iter([
        '<tool_call>{"name":"turn_light","arguments":{"room":"kitchen","state":"on"}}</tool_call>'
        '<tool_call>{"name":"set_temperature","arguments":{"room":"bedroom","value":40}}</tool_call>',
        "Done",
    ])
    functions["generate"] = lambda *args: next(outputs)
    home = functions["SmartHome"]()
    messages = [{"role": "user", "content": "test"}]
    functions["run_agent"](None, None, home, messages)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert len(messages[1]["tool_calls"]) == 2
    assert "error" in json.loads(messages[3]["content"])
    assert home.rooms["kitchen"]["light"] == "on"
    assert home.rooms["bedroom"]["temperature"] == 21


def test_agent_stops_at_limit_and_keeps_malformed_reply():
    functions = load_functions()
    functions["generate"] = lambda *args: '<tool_call>{"name":"get_room_state","arguments":{"room":"kitchen"}}</tool_call>'
    messages = []
    functions["run_agent"](None, None, functions["SmartHome"](), messages, max_steps=2)
    assert len(messages) == 4
    functions["generate"] = lambda *args: "<tool_call>broken"
    messages = []
    functions["run_agent"](None, None, functions["SmartHome"](), messages)
    assert messages == [{"role": "assistant", "content": "<tool_call>broken"}]


def test_relative_temperature_uses_actual_state_across_user_turns():
    functions = load_functions()

    def respond(model, tokenizer, messages, tools):
        latest = messages[-1]
        if latest["role"] == "user":
            call = {"name": "get_room_state", "arguments": {"room": "bedroom"}}
        elif latest["name"] == "get_room_state":
            state = json.loads(latest["content"])
            call = {"name": "set_temperature", "arguments": {
                "room": "bedroom", "value": state["temperature"] + 2,
            }}
        else:
            return "Done"
        return "<tool_call>" + json.dumps(call) + "</tool_call>"

    functions["generate"] = respond
    home = functions["SmartHome"]()
    messages = []
    for prompt in ("Make it two degrees warmer in the bedroom.", "Two more degrees."):
        messages.append({"role": "user", "content": prompt})
        functions["run_agent"](None, None, home, messages)
    temperatures = [json.loads(message["content"])["temperature"]
                    for message in messages if message["role"] == "tool"]
    assert temperatures == [21, 23, 23, 25]
    assert home.rooms["bedroom"]["temperature"] == 25
    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ] * 2
