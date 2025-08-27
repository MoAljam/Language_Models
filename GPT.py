from typing import Optional
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigGPT:
    vocab_size: int = 256
    block_size: int = 128  # context length T
    n_embed: int = 512
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0
    bias: bool = False


class SimpleMLP(nn.Module):
    def __init__(self, config: ConfigGPT):
        super(SimpleMLP, self).__init__()
        self.lin_wide = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin_proj_back = nn.Linear(config.n_embed * 4, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_wide(x)
        x = self.gelu(x)
        x = self.lin_proj_back(x)
        x = self.dropout(x)
        return x


# region causal self-attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ConfigGPT):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embed % config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.head_dim = config.n_embed // config.n_head

        # all key, query, value projections combined, split into individual vectors and heads later
        self.self_attn = nn.Linear(config.n_embed, config.n_embed * 3, bias=config.bias)
        self.dropout_attn = nn.Dropout(config.dropout)

        # project back to n_mbed for the output of the attention heads
        self.lin_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.dropout_proj = nn.Dropout(config.dropout)

        # tril mask for decoder self-attention (masking future tokens)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.flash_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # compute key, query, value projections
        # self attention layer holds currently all k, q , v off all heads in one tensor in (B, T, 3 * n_embd)
        # make it explicit                      ->  (B, T, n_head, 3, head_dim)
        # transpose to to pull head before T    ->  (B, n_head, T, 3, head_dim)
        qkv = self.self_attn(x)  # (B, T, 3 * n_embd)
        qkv = qkv.view(B, T, self.n_head, 3, C // self.n_head)  # (B, T, n_head, 3, head_dim)
        qkv = qkv.transpose(1, 2)  # (B, n_head, T, 3, head_dim)
        # print(f"qkv shape: {qkv.shape}")

        # split into q, k, v
        q, k, v = qkv.unbind(dim=3)  # (B, n_head, T, head_dim)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        # use flash attention if available
        if self.flash_attn:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                # attn_mask=self.mask[:, :, :T, :T],
            )
        else:
            # compute affinity matrix with scaled dot-product attention from original transformer paper
            # aff = (Q @ K^T) / sqrt(d_k)
            aff = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, n_head, T, T)
            # apply mask to the attention matrix
            aff = aff.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # (B, n_head, T, T)
            # softmax to get attention weights
            aff = F.softmax(aff, dim=-1)  # (B, n_head, T, T)
            # apply dropout to attention weights
            aff = self.dropout_attn(aff) if self.training else aff  # (B, n_head, T, T)
            # compute output of attention heads
            # out = aff @ v  , (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
            out = aff @ v  # (B, n_head, T, head_dim)

        # flatten heads and concatenate them
        # here view could fail, reshape is more robust (handles non-contiguous automatically)
        # out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, n_head * head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_head * head_dim) -> (B, T, C)
        # print(f"out shape: {out.shape}")

        # project back to n_embed
        out = self.lin_proj(out)  # (B, T, C)
        out = self.dropout_proj(out) if self.training else out  # (B, T, C)

        return out  # (B, T, C)


class DecoderBlock(nn.Module):
    def __init__(self, config: ConfigGPT):
        super(DecoderBlock, self).__init__()
        self.n_embed = config.n_embed

        self.layer_norm_1 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        # self.layer_norm_1 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.layer_norm_mlp = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = SimpleMLP(config)
        # self.layer_norm_mlp = nn.LayerNorm(config.n_embed, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x + self.layer_norm_1(self.attention(x))
        # x = x + self.layer_norm_mlp(self.mlp(x))
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_mlp(x))
        return x  # (B, T, C)


# region gpt
class GPT(nn.Module):
    def __init__(self, config: ConfigGPT):
        super(GPT, self).__init__()
        # self.vocab_size = config.vocab_size
        # self.block_size = config.block_size
        # self.n_embed = config.n_embed
        # self.n_head = config.n_head
        # self.n_layer = config.n_layer

        for k, v in config.__dict__.items():
            setattr(self, k, v)

        # token empedding
        # position embedding
        # dropout for combined embeddings (token + position)
        # n layers of transformer blocks
        # ---- res and layer norm
        # ---- masked multi-head attention (causal self-attention)
        # -------- linear for query, key, value ( one layer or three separate layers )
        # -------- split into heads (either as different or individual tensors)
        # -------- scaled dot-product attention
        # ------------ q @ k^T / sqrt(d_k)
        # ------------ softmax
        # ------------ dropout
        # ------------ v @ softmax(q @ k^T / sqrt(d_k))
        # -------- concat heads
        # -------- linear output layer
        # -------- dropout
        # ---- res and layer norm
        # ---- mlp (ffn)
        # layer norm
        # linear output layer

        self.embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.dropout_em = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)])
        self.lyer_norm = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.lin_out = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)

        # initialize weights
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, only_last=False) -> torch.Tensor:
        B, T = x.size()
        device = x.device
        assert T <= self.block_size, "Input sequence length exceeds block size."

        # get token and position embeddings
        token_emb = self.embedding(x)  # (B, T, n_embed)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)
        combined_emb = token_emb + pos_emb
        x = self.dropout_em(combined_emb)  # (B, T, n_embed)

        # pass through decoder blocks
        x = self.decoder_blocks(x)  # (B, T, n_embed)
        x = self.lyer_norm(x)
        x = self.lin_out(x)  # (B, T, vocab_size)
        if only_last:
            return x[:, [-1], :]  # NOTE: returns only last token's logits (B, 1, vocab_size)
        return x  # (B, T, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_0(
        self,
        input_ids: torch.Tensor,  # (B, T_in) long
        *,
        max_new_tokens: int = 50,
        mode: str = "sample",  # sample or argmax
        eos_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,  # e.g. 50
        top_p: Optional[float] = None,  # e.g. 0.9
        repetition_penalty: float = 1.0,  # >1.0 discourages repeats
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate text from the model.

        Parameters:
        - input_ids: Tensor of shape (B, T_in) with input token IDs.
        - max_new_tokens: Maximum number of new tokens to generate.
        - mode: 'sample' for sampling, 'argmax' for greedy decoding.
        - eos_id: Optional end-of-sequence token ID.
        - temperature: Temperature for sampling.
        - top_k: Top-k filtering for sampling.
        - top_p: Top-p (nucleus) filtering for sampling.
        - repetition_penalty: Penalty factor for repeated tokens.
        - device: Device to run the generation on.

        Returns:
        - Generated token IDs of shape (B, T_out).
        """
        assert input_ids.dtype == torch.long, "input_ids must be LongTensor (token IDs)."
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device

        x = input_ids.to(device)
        B = x.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            # context: last token only (shape (B, 1))
            last_tok = x[:, -1:].to(device)
            # logits: (B, 1, V) -> take last step -> (B, V)
            logits = self(last_tok)[:, -1, :]
            # repetition penalty (simple): down-weight tokens already present
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(B):
                    if not finished[b]:
                        seen = torch.unique(x[b])
                        logits[b, seen] -= math.log(repetition_penalty)

            # greedy argmax
            if mode == "argmax":
                next_ids = torch.argmax(logits, dim=-1)
            # else sample according after applying temperature, top-k, top-p
            else:
                # temperature scaling
                if temperature is not None and temperature != 1.0:
                    logits = logits / max(temperature, 1e-8)

                # top-k filter
                if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                    kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

                # top-p filter
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)

                    # mask tokens with cumulative prob > top_p
                    mask = cumprobs > top_p
                    # ensure at least one token remains
                    mask[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    # unsort back
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # If EOS provided, keep it once reached
            if eos_id is not None:
                next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)

            # append
            x = torch.cat([x, next_ids.unsqueeze(1)], dim=1)

            # update finished batch mask
            if eos_id is not None:
                finished = finished | (next_ids == eos_id)
                if torch.all(finished):
                    break

        if was_training:
            self.train()
        return x


if __name__ == "__main__":

    # test components
    config = ConfigGPT()
    test_input = torch.randn(2, config.block_size, config.n_embed)  # (B, T, C)
    print(f"config: {config}")
    print(f"test_input shape: {test_input.shape}")  # should be (B, T, C)

    csa = CausalSelfAttention(config)
    output = csa(test_input)
    print(f"output shape: {output.shape}")  # should be (B, T, C)

    mlp = SimpleMLP(config)
    output_mlp = mlp(test_input)
    print(f"output mlp shape: {output_mlp.shape}")  # should be (B, T, C)

    block = DecoderBlock(config)
    output_block = block(test_input)
    print(f"output block shape: {output_block.shape}")  # should be (B, T, C)

    GPT_model = GPT(config)  # ensure input is within block size
    test_input_gpt = torch.randint(0, config.vocab_size, (2, config.block_size))  # (B, T)
    output_gpt = GPT_model(test_input_gpt)
    print(f"output GPT shape: {output_gpt.shape}")  # should be (B, T, vocab_size)
