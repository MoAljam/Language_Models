import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import dataclasses


@dataclasses.dataclass
class ConfigNeuralBigram:
    vocab_size: int = 256
    dropout: float = 0.0


class NeuralBigram(nn.Module):
    def __init__(self, config: ConfigNeuralBigram):
        super(NeuralBigram, self).__init__()
        self.vocab_size = config.vocab_size
        self.dropout_rate = config.dropout
        self.embedding = nn.Embedding(config.vocab_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x

    @torch.no_grad()
    def generate(
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

    @torch.no_grad()
    def generate_0(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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
