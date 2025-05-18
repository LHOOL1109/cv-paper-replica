import torch
import torch.nn as nn
from torch import Tensor


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_stacks: int = 8, ffn_dim: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, ffn_dim) for _ in range(num_stacks)])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, ffn_dim: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attn_out = self.attention(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        d_k = k.shape[-1]
        score = (q @ k.transpose(-2, -1)) / d_k ** 0.5
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        prob = self.softmax(score)
        return prob @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.attention = ScaleDotProductAttention()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        out: Tensor = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 512):
        super().__init__()
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encoding[:, :x.size(1)]


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int = 512):
        super().__init__(vocab_size, d_model, padding_idx=1)
