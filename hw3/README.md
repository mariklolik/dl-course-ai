# ДЗ3: SFT для function calling

[Notebook](sft_function_calling_hw.ipynb): генерация, маска loss, полное SFT, оценка и агентный цикл.

Четыре CPU-проверки проходят; generation-теги сохраняют исходный текст всех 922 обучающих и валидационных диалогов.

Обучение и оценка пока не выполнены: Slurm не выделил GPU на FSK.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab sft_function_calling_hw.ipynb
```

[Статус проверок и хеши данных](outputs/verification.json)
