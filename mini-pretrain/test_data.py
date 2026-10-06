import importlib.util
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


spec = importlib.util.spec_from_file_location("solution", Path(__file__).with_name("your_solution.py"))
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def test_padding_is_not_a_target_and_real_eos_is_preserved():
    tokenizer = AutoTokenizer.from_pretrained(solution.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    result = solution.tokenize_function({"text": ["Москва" + tokenizer.eos_token, ""]}, tokenizer)
    for ids, mask, labels in zip(result["input_ids"], result["attention_mask"], result["labels"]):
        assert len(ids) == len(mask) == len(labels) == 512
        assert labels == [token if active else -100 for token, active in zip(ids, mask)]
    assert tokenizer.eos_token_id in result["labels"][0]
    assert result["labels"][1] == [-100] * 512


def test_parquet_roundtrip_preserves_holdout_order(tmp_path):
    dataset = Dataset.from_dict({"row": list(range(5007))})
    solution.save_as_parquets(dataset, str(tmp_path), num_shards=4)
    restored = solution.load_tokenized_dataset(str(tmp_path))
    train, evaluation = solution.split_dataset(restored)
    assert evaluation["row"] == list(range(5000))
    assert train["row"] == list(range(5000, 5007))


def test_timeout_is_measured_from_training_start(monkeypatch, tmp_path):
    from types import SimpleNamespace

    now = [100.0]
    monkeypatch.setattr(solution.time, "monotonic", lambda: now[0])
    callback = solution.TimeoutCallback(900, tmp_path)
    control = SimpleNamespace(should_training_stop=False, should_evaluate=False, should_save=False)
    callback.on_train_begin(None, None, control)
    now[0] = 999.0
    callback.on_step_end(None, None, control)
    assert not control.should_training_stop
    now[0] = 1001.0
    callback.on_step_end(None, None, control)
    assert control.should_training_stop and control.should_evaluate and control.should_save
