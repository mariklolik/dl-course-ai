import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        super(LanguageModel, self).__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(indices.clamp(min=0))
        output, _ = self.rnn(embedded)
        logits = self.linear(output)
        return logits[:, :lengths.max(), :]

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        self.eval()
        device = next(self.parameters()).device
        bos = self.dataset.bos_id
        eos = self.dataset.eos_id

        tokens = [bos] + self.dataset.text2ids(prefix)
        input_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        embedded = self.embedding(input_tensor)
        _, hidden = self.rnn(embedded)

        generated = list(tokens[1:])
        last_token = tokens[-1]

        for _ in range(self.max_length):
            x = torch.tensor([[last_token]], dtype=torch.long, device=device)
            emb = self.embedding(x)
            out, hidden = self.rnn(emb, hidden)
            logits = self.linear(out.squeeze(1))
            token = torch.multinomial(torch.softmax(logits / temp, dim=-1), 1).item()
            if token == eos:
                break
            generated.append(token)
            last_token = token

        return self.dataset.ids2text(generated)
