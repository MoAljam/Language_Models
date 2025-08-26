from collections import Counter
from scipy.special import softmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

# region NGramEngine


class NGramEngine:
    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.vocab = None
        self.total_vocab_size = None
        self.lambdas = None
        self.pad_token = None

    def fit(self, text, lambdas=None, pad_token=None):
        """
        Fit the n-gram model to the provided text.

        Parameters:
        - text: str, list or tuple of tokens
            The text to fit the model on. If a string, it will be split into tokens.
        - lambdas: list or np.ndarray, optional
            Smoothing parameters for each n-gram level (length must be n).
        - pad_token: any, optional
            Token to use for padding shorter n-grams.
        """
        if isinstance(text, str):
            self.vocab = tuple(set(text.split()))
            total_tokens_count = len(text.split())
        elif isinstance(text, list) or isinstance(text, tuple):
            self.vocab = tuple(set(text))
            total_tokens_count = len(text)
        else:
            raise ValueError("text must be a string, a list or a tuple of tokens")

        self.total_vocab_size = len(self.vocab)

        # self.counts[0] = Counter({():self.total_vocab_size}) # NOTE total vocab size or total token count
        self.counts[0] = Counter({(): total_tokens_count})  # NOTE total vocab size or total token count

        for i in range(1, self.n + 1):
            self.counts[i] = self.n_gram_counts(self.get_n_grams(text, i))

        self.lambdas = np.asarray(lambdas) if lambdas is not None else np.ones(self.n) / self.n

        if np.isclose(np.sum(self.lambdas), 1.0, atol=1e-3) is False:
            raise ValueError(f"lambdas must sum to 1, got {np.sum(self.lambdas)}")
        if len(self.lambdas) != self.n:
            raise ValueError(f"lambdas must have length {self.n}, got {len(self.lambdas)}")

        self.pad_token = pad_token if pad_token is not None else 0
        self.fitted = True

    def get_n_grams(self, text, n=None):
        """
        Generate n-grams from the provided text.

        Parameters:
        - text: str, list or tuple of tokens
            The text to generate n-grams from. If a string, it will be split into tokens.
        - n: int, optional
            The size of the n-grams to generate. If None, uses the model's n.

        Returns:
        - list of tuples representing n-grams.
        """
        if n is None:
            n = self.n
        if isinstance(text, str):
            text = text.split()
        if n > self.n:
            raise ValueError(f"n ({n}) cannot be greater than the model's n ({self.n})")

        return [tuple(text[i : i + n]) for i in range(len(text) - n + 1)]

    def n_gram_counts(self, n_grams):
        return Counter(tuple(n_gram) for n_gram in n_grams)

    def _n_gram_prob(self, n_gram_query, n, k_smoothing=1, verbose=False):
        """
        Calculate the probability of an n-gram query using the n-gram counts.

        Parameters:
        - n_gram_query: tuple
            The n-gram query to calculate the probability for.
        - n: int
            The size of the n-gram.
        - k_smoothing: int, optional
            Smoothing parameter to avoid zero probabilities.
        - verbose: bool, optional
            If True, prints detailed information about the probability calculation.

        Returns:
        - float: The probability of the n-gram query.
        """
        if not self.fitted:
            raise ValueError("ngram moodel must be fitted before calculating probabilities.")

        if n > self.n:
            raise ValueError(f"n ({n}) cannot be greater than the model's n ({self.n})")

        n_gram_counts = self.counts[n]
        n_minus_one_gramm_counts = self.counts[n - 1]

        num = n_gram_counts.get(n_gram_query, 0) + k_smoothing
        denom = n_minus_one_gramm_counts.get(n_gram_query[:-1], 0) + k_smoothing * self.total_vocab_size

        prob = num / denom if denom != 0 else 0

        if verbose:
            print(
                f"Query: {n_gram_query}",
                f"Prefix:{n_gram_query[:-1]}",
                f"N={n}, Num: {num}, Denom: {denom}, Prob: {prob}",
                sep="\n",
            )
        return prob

    def get_n_gram_prob(self, n_gram_query, interpolate=True, n=None, get_all_probs=False, **kwargs):
        """
        Get the probability of an n-gram query, optionally using interpolation.

        Parameters:
        - n_gram_query: tuple or str
            The n-gram query to calculate the probability for. If a string, it will be split into tokens.
        - interpolate: bool, optional
            If True, uses interpolation with lower n-grams.
        - n: int, optional
            The size of the n-gram. If None, uses the model's n.
        - get_all_probs: bool, optional
            If True, returns probabilities for all n-grams up to n.
        - **kwargs: additional keyword arguments for flexibility (e.g., k_smoothing).

        Returns:
        - float or tuple: The probability of the n-gram query or a tuple of (probability, all_probs).
        """
        if isinstance(n_gram_query, str):
            n_gram_query = tuple(n_gram_query.split())
        elif isinstance(n_gram_query, list):
            n_gram_query = tuple(n_gram_query)
        if n is None:
            n = self.n
        if n > self.n:
            raise ValueError(f"n ({n}) cannot be greater than the model's n ({self.n})")

        # n = len(n_gram_query) if len(n_gram_query) < n else n
        if len(n_gram_query) < n:
            # padding for shorter n-grams
            n_gram_query = n_gram_query + (self.pad_token,) * (n - len(n_gram_query))
        elif len(n_gram_query) > n:
            # truncate longer n-grams
            n_gram_query = n_gram_query[-n:]

        # when using lower n-grams with interpolation, adjust the lambdas
        lambdas = self.lambdas.copy()
        if n != self.n:
            lambdas = lambdas[:n]
            lambdas /= np.sum(lambdas)

        if interpolate:
            probs = np.zeros(n)
            for m in range(n, 0, -1):
                query = n_gram_query[-m:]
                prob = self._n_gram_prob(query, m, **kwargs)
                probs[m - 1] = prob
            prob = np.dot(probs, lambdas)
            if get_all_probs:
                return prob, probs
            return prob

        prob = self._n_gram_prob(n_gram_query, n, **kwargs)
        return prob

    def get_next_token_probs(self, context, n=None, interpolate=True, **kwargs):
        """
        Get the probabilities of the next token given a context.

        Parameters:
        - context: str, list or tuple
            The context for which to calculate the next token probabilities. If a string, it will be split into tokens.
        - n: int, optional
            The size of the n-gram. If None, uses the model's n.
        - interpolate: bool, optional
            If True, uses interpolation with lower n-grams.
        - **kwargs: additional keyword arguments for flexibility (e.g., k_smoothing, get_all_probs).

        Returns:
        - tuple: (probabilities of next tokens, list of next tokens)
        """
        if isinstance(context, str):
            context = tuple(context.split())
        elif isinstance(context, list):
            context = tuple(context)

        N = self.n if n is None else n

        if N > self.n:
            raise ValueError(f"n ({n}) cannot be greater than the model's n ({self.n})")

        if N == 1:
            context = tuple()

        # get the probability of all possible next tokens
        probs = np.zeros(len(self.vocab))
        keys = []
        next_token_all_probs = np.zeros((len(self.vocab), N))
        for idx, token in enumerate(self.vocab):
            n_gram_query = (*context, token)
            if kwargs.get("get_all_probs", False):  # just for playing around
                prob, all_probs = self.get_n_gram_prob(n_gram_query, interpolate, N, **kwargs)
                next_token_all_probs[idx, :] = all_probs
            else:
                prob = self.get_n_gram_prob(n_gram_query, interpolate, N, **kwargs)
            keys.append(token)
            probs[idx] = prob

        if kwargs.get("get_all_probs", False):
            return probs, keys, next_token_all_probs
        return probs, keys

    def fit_metrics(self, tokens, n=None, interpolate=True, verbose=False, **kwargs):
        """
        Calculate perplexity, bits per token, and nats per token for the given tokens.

        Parameters:
        - tokens: str, list or tuple
            The tokens to calculate metrics for. If a string, it will be split into tokens.
        - n: int, optional
            The size of the n-gram. If None, uses the model's n.
        - interpolate: bool, optional
            If True, uses interpolation with lower n-grams.
        - verbose: bool, optional
            If True, prints detailed information about the calculation.
        - **kwargs: additional keyword arguments for flexibility (e.g., k_smoothing).

        Returns:
        - dict: containing perplexity (ppl), bits per token (bpt), nats per token (nats_per_tok),
                 effective tokens used in calculation, and total number of tokens.
        """
        if isinstance(tokens, str):
            tokens = tokens.split()
        elif isinstance(tokens, list):
            tokens = tuple(tokens)

        if n is None:
            n = self.n

        if n > self.n:
            raise ValueError(f"n ({n}) cannot be greater than the model's n ({self.n})")

        log_prob_sum = 0
        # effective_tokens = len(text) # if padding used
        effective_tokens = len(tokens) - n + 1

        eps = 1e-12
        # for i in range(len(text)): # if padding used
        for i in range(n - 1, len(tokens)):
            # if padding used
            # if i < n - 1:
            #     context = (self.pad_token,) * (n - 1 - i) + text[:i]
            # else:
            #     context = text[i - n + 1 : i]

            context = tokens[i - n + 1 : i]
            token = tokens[i]
            n_gram = tuple(context) + (token,)
            prob = self.get_n_gram_prob(n_gram, interpolate=interpolate, n=n, **kwargs)

            # log_prob = np.log(prob if prob > 0 else 1e-12)
            log_prob = np.log(max(prob, eps))
            log_prob_sum += log_prob
            if verbose:
                print(f"Context: {context}, Token: {token}\nProb: {prob:.6f}, Log Prob: {log_prob:.6f}")

        nats_per_tok = -log_prob_sum / effective_tokens if effective_tokens > 0 else float("inf")
        bpt = nats_per_tok / np.log(2)
        ppl = np.exp(nats_per_tok) if effective_tokens > 0 else float("inf")

        return {
            "ppl": ppl,
            "bpt": bpt,
            "nats_per_tok": nats_per_tok,
            "effective_tokens": effective_tokens,
            "num_tokens": len(tokens),
        }

    def generate(
        self,
        context,
        max_new_tokens: int = 30,
        *,
        n: int | None = None,
        mode: str = "argmax",  # "argmax" or "sample"
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,  # <1.0 discourages repeats
        stop_tokens: list | None = None,  # list of ids or tokens
        ban_tokens: list | None = None,  # ids/tokens to never emit (e.g., BOS)
        verbose: bool = False,
        **kwargs,
    ):
        """
        Generate text based on the context using n-gram probabilities.

        Parameters:
        - context: str, list or tuple of tokens
            The context to start generating from. If a string, it will be split into tokens.
        - max_len: int, optional
            The maximum length of the generated text.
        - n: int, optional
            The size of the n-gram. If None, uses the model's n.
        - mode: str, optional
            "argmax" for greedy generation or "sample" for sampling.
        - temperature: float, optional
            Temperature for sampling (1.0 is no change).
        - top_k: int, optional
            Top-k filtering for sampling.
        - top_p: float, optional
            Top-p (nucleus) filtering for sampling.
        - repetition_penalty: float, optional
            Penalty for repeating tokens (less than 1.0 discourages repeats).
        - stop_tokens: list, optional
            List of tokens to stop generation when encountered.
        - ban_tokens: list, optional
            List of tokens to never emit (e.g., BOS).
        - verbose: bool, optional
            If True, prints detailed information about the generation.
        - **kwargs: additional keyword arguments for flexibility (e.g., k_smoothing).

        Returns:
        - list: generated tokens as a list.
        """
        if isinstance(context, str):
            # assume space-separated ids if str
            context = context.split()
        elif isinstance(context, tuple):
            context = list(context)
        else:
            context = list(context)

        N = self.n if n is None else n
        if N > self.n:
            raise ValueError(f"n ({N}) cannot be greater than model's n ({self.n})")

        # # Map stop/ban lists to the same type as vocab elements
        # vocab_list = list(self.vocab)
        # # Build quick index for token -> position in vocab_list
        # tok2idx = {t: i for i, t in enumerate(vocab_list)}

        # def _to_idx_list(tokens_maybe):
        #     if tokens_maybe is None:
        #         return None
        #     out = []
        #     for t in tokens_maybe:
        #         out.append(tok2idx.get(t, t) if t in tok2idx else t)  # if already id, keep
        #     return out

        # stop_idx = set(_to_idx_list(stop_tokens) or [])
        # ban_idx = set(_to_idx_list(ban_tokens) or [])
        stop_tokens = set(stop_tokens) if stop_tokens is not None else set()
        ban_tokens = set(ban_tokens) if ban_tokens is not None else set()

        if verbose:
            print(
                f"Generating with n={N}, mode={mode}, T={temperature}, top_k={top_k}, top_p={top_p}, ",
                f"rep_pen={repetition_penalty}",
            )

        for _ in range(max_new_tokens):
            # Get next-token probabilities (not guaranteed to sum to 1; we renormalize downstream)
            probs, tokens = self.get_next_token_probs(context, n=N, interpolate=True, **kwargs)
            probs = np.asarray(probs, dtype=np.float64)

            # Build masks
            banned_mask = np.zeros(len(tokens), dtype=bool)
            if ban_tokens:
                for i, t in enumerate(tokens):
                    # t is a vocab element; map to index space for comparison
                    if t in ban_tokens:
                        banned_mask[i] = True

            # Repetition mask: tokens already present in context
            seen_set = set(context)
            seen_mask = np.array([(t in seen_set) for t in tokens], dtype=bool)

            if mode == "argmax":
                # Greedy on renormalized probs after banning & repetition penalty (if <1)
                p = probs.copy()
                if banned_mask.any():
                    p[banned_mask] = 0.0
                if repetition_penalty < 1.0:
                    p[seen_mask] *= max(repetition_penalty, 0.0)
                # Renormalize safely
                s = p.sum()
                if s <= 0:
                    # fallback: choose non-banned token deterministically
                    candidates = np.where(~banned_mask)[0]
                    next_idx = int(candidates[0] if candidates.size > 0 else np.argmax(probs))
                else:
                    p /= s
                    next_idx = int(np.argmax(p))
            elif mode == "sample":
                next_idx = _filter_and_sample(
                    probs,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seen_mask=seen_mask,
                    banned_mask=banned_mask,
                )
            else:
                raise ValueError("mode must be 'argmax' or 'sample'")

            next_token = tokens[next_idx]
            context.append(next_token)

            # stopping
            if stop_tokens and next_token in stop_tokens:
                break

        return context


# region UTILS


def check_context_distribution(model, context, k_smoothing=1, top_k=10, interpolate=True, n=None):
    probs, tokens = model.get_next_token_probs(
        context, interpolate=interpolate, k_smoothing=k_smoothing, n=n, get_all_probs=False
    )
    total_sum = np.sum(probs)

    print(f"\nContext: {context}")
    print(f"Sum of probs = {total_sum:.6f}")
    print(f"Top {top_k} tokens:")
    for t, p in sorted(zip(tokens, probs), key=lambda x: -x[1])[:top_k]:
        print(f"  {t:<10} {p:.6f}")

    return dict(zip(tokens, probs))


def _filter_and_sample(
    probs: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    seen_mask: np.ndarray | None = None,
    banned_mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> int:
    p = probs.astype(np.float64).copy()

    # zero-out banned tokens
    if banned_mask is not None:
        p[banned_mask] = 0.0

    # apply repetition penalty on seen tokens
    if repetition_penalty and seen_mask is not None:
        p[seen_mask] *= max(repetition_penalty, 0.0)

    # early exit if all zeros -> fallback to uniform over non-banned
    if p.sum() <= 0:
        # pick any non-banned index
        if banned_mask is not None and (~banned_mask).any():
            candidates = np.flatnonzero(~banned_mask)
            return int(np.random.choice(candidates))
        # otherwise argmax of original
        return int(np.argmax(probs))

    # temperature on probabilities (monotone, preserves ranking)
    if temperature is not None and temperature > 0.0:
        p = np.power(p, 1.0 / temperature)

    # renormalize before top-k/p
    total = p.sum()
    if total > 0:
        p /= total
    else:
        p = np.ones_like(p) / len(p)

    # top-k filter
    if top_k is not None and top_k > 0 and top_k < p.size:
        kth = np.partition(p, -top_k)[-top_k]
        mask = p < kth
        p[mask] = 0.0
        s = p.sum()
        p = p / s if s > 0 else np.ones_like(p) / p.size

    # top-p filter
    if top_p is not None and 0.0 < top_p < 1.0:
        idx_sorted = np.argsort(-p)  # descending sort
        p_sorted = p[idx_sorted]
        cdf = np.cumsum(p_sorted)
        keep = cdf <= top_p
        # always keep at least one token
        if not keep.any():
            keep[0] = True
        mask_sorted = ~keep
        p_sorted[mask_sorted] = 0.0
        # unsort back
        p = np.zeros_like(p)
        p[idx_sorted] = p_sorted
        s = p.sum()
        p = p / s if s > 0 else np.ones_like(p) / p.size

    chosen_idx = int(np.random.choice(len(p), p=p))
    return chosen_idx


def perplexity_over_n(n, train_ids, val_ids, pad_token_id=None, verbose=False):
    interpolate = False
    results = {
        "n": [],
        "ppl_val": [],
        "bpt_val": [],
        "nats_per_tok_val": [],
        "ppl_train": [],
        "bpt_train": [],
        "nats_per_tok_train": [],
        "num_n_grams": [],
    }
    for n in range(1, n):

        model = NGramEngine(n)
        model.fit(train_ids, pad_token=pad_token_id)

        metrics_train = model.fit_metrics(train_ids, interpolate=interpolate)
        metrics_val = model.fit_metrics(val_ids, interpolate=interpolate)
        t_ppl, t_bpt, t_nats_per_tok = metrics_train["ppl"], metrics_train["bpt"], metrics_train["nats_per_tok"]
        v_ppl, v_bpt, v_nats_per_tok = metrics_val["ppl"], metrics_val["bpt"], metrics_val["nats_per_tok"]

        results["n"].append(n)
        results["ppl_val"].append(v_ppl)
        results["bpt_val"].append(v_bpt)
        results["nats_per_tok_val"].append(v_nats_per_tok)
        results["ppl_train"].append(t_ppl)
        results["bpt_train"].append(t_bpt)
        results["nats_per_tok_train"].append(t_nats_per_tok)
        results["num_n_grams"].append(len(model.counts[n]))

        if verbose:
            print(f"Training N-gram model with n={n}, lambdas={model.lambdas}")
            print("model-vocab size:", model.total_vocab_size)
            print("prob of non-existened n-gram and (n-1)-gram :", 1 / model.total_vocab_size)  # 1/vocab_siz
            print("num n_grams: ", len(model.counts[n]))
            print("most common:\n", model.counts[n].most_common(5))
            print("#" * 80)
    return results


# region PLOTTING


def plot_ngram_probs(n_gram_probs, top_n=20, prob_map=lambda x: x, title="", ax=None):
    df = pd.DataFrame(n_gram_probs.items(), columns=["token", "prob"])
    # df['normalized_prob'] = softmax(df['prob'])
    df["normalized_prob"] = df["prob"]
    df["normalized_prob"] = prob_map(df["normalized_prob"])
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    print(f"sum of probabilities for {len(df['token'][0])}-gram: ", df["normalized_prob"].sum())

    show_plot = False
    if ax is None:
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        show_plot = True

    ax.plot(range(len(df.head(top_n))), df["normalized_prob"].head(top_n))

    ax.set_xticks(range(len(df.head(top_n))))
    ax.set_xticklabels(df["token"].head(top_n), rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("mapped P(token)")

    # plot average probability line with a dashed line
    avg_prob = np.mean(df["normalized_prob"].head(top_n))
    overall_avg_prob = np.mean(df["normalized_prob"])
    ax.axhline(y=avg_prob, color="r", linestyle="--", label=f"Average top {top_n}\noverall:{overall_avg_prob:.6f}")

    # add legend
    ax.legend()

    if show_plot:
        plt.tight_layout()
        plt.draw()

    return ax


def plot_perplexity_and_n_grams_combinations(ppl_results, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ppl_results["n"], ppl_results["ppl_train"], marker="o", label="Train PPL")
    plt.plot(ppl_results["n"], ppl_results["ppl_val"], marker="o", label="Validation PPL")
    plt.xticks(ppl_results["n"])
    plt.xlabel("(n)-gram")
    plt.ylabel("Perplexity")
    plt.title("Perplexity on validation and train sets")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        ppl_results["n"], ppl_results["num_n_grams"], marker="*", color="dimgray", label="Number of unique N-grams"
    )
    plt.xticks(ppl_results["n"])
    plt.xlabel("(n)-gram")
    plt.ylabel("Number of unique N-grams")
    plt.title("unique n-grams for different n-gram sizes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
