from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD, Lowercase, Sequence
from tokenizers.processors import TemplateProcessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

import os
from typing import List, Callable, Dict, Any

TOK_SAVE_DIR = "../bpe_tok"
TOK_SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}


def train_bytelevel_bpe(
    files: List[str],
    merges: int = 2000,
    *,
    min_frequency: int = 2,
    lowercase: bool = False,
    add_prefix_space: bool = True,
    save_dir: str = TOK_SAVE_DIR,
    special_tokens: dict = TOK_SPECIAL_TOKENS,
    initial_alphabet: List[str] = ByteLevel.alphabet(),
    save_filename: str = None,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on the provided text files.

    Parameters:
    - files: List of text files to train on.
    - merges: Number of BPE merges to perform.
    - min_frequency: Minimum frequency for a token to be included in the vocabulary.
    - lowercase: Whether to lowercase the input text.
    - add_prefix_space: Whether to add a space before each byte (GPT-style).
    - save_dir: Directory to save the trained tokenizer.
    - name: Name for the saved tokenizer file.
    - special_tokens: Dictionary of special tokens to include in the vocabulary.
    - initial_alphabet: Initial alphabet for byte-level BPE.

    Returns:
    - Trained Tokenizer instance.
    """
    unk_token = special_tokens.get("unk", TOK_SPECIAL_TOKENS["unk"])
    bos_token = special_tokens.get("bos", TOK_SPECIAL_TOKENS["bos"])
    eos_token = special_tokens.get("eos", TOK_SPECIAL_TOKENS["eos"])

    # For byte-level BPE, base symbols = 256 bytes + special tokens
    base_vocab = 256 + len(special_tokens)
    vocab_size = base_vocab + merges

    tok = Tokenizer(BPE(unk_token=unk_token))

    # Normalization (optional)
    norms = [NFD()]
    if lowercase:
        norms.append(Lowercase())
    tok.normalizer = Sequence(norms)

    # GPT-style byte-level pretokenizer & decoder
    tok.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens.values()),
        initial_alphabet=initial_alphabet,  # ensure 256 bytes are present
        show_progress=False,
    )

    # Train on your text files (can pass many)
    tok.train(files=files, trainer=trainer)

    # Optional: auto add BOS/EOS around sequences
    tok.post_processor = TemplateProcessing(
        single=f"{bos_token} $0 {eos_token}",
        pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[
            (bos_token, tok.token_to_id(bos_token)),
            (eos_token, tok.token_to_id(eos_token)),
        ],
    )

    if save_filename is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{save_filename}")
        tok.save(path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def train_and_encode_tokenizer(
    *,
    tokenizer_trainer,  # e.g. train_bytelevel_bpe
    train_text_path: str,
    other_texts_paths: dict[str:str] = None,
    merges: int = 200,
    **kwargs,
) -> Dict[str, Any]:

    if other_texts_paths is None:
        other_texts_paths = {}

    with open(train_text_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    other_texts = {}
    for key, path in other_texts_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            other_texts[key] = f.read()

    tok = tokenizer_trainer(
        files=[train_text_path],
        merges=merges,
        **kwargs,
    )
    encode = partial(tok.encode, add_special_tokens=False)

    pad_token = kwargs.get("special_tokens", TOK_SPECIAL_TOKENS).get("pad", "<pad>")
    bos_token = kwargs.get("special_tokens", TOK_SPECIAL_TOKENS).get("bos", "<bos>")
    eos_token = kwargs.get("special_tokens", TOK_SPECIAL_TOKENS).get("eos", "<eos>")
    pad_id = tok.token_to_id(pad_token)
    bos_id = tok.token_to_id(bos_token)
    eos_id = tok.token_to_id(eos_token)

    train_ids = encode(train_text).ids
    other_texts_ids = {}
    for key, text in other_texts.items():
        other_texts_ids[key] = encode(text).ids

    return {
        "tokenizer": tok,
        "vocab_size": tok.get_vocab_size(),
        "train_ids": train_ids,
        "other_texts_ids": other_texts_ids,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }


# region utils


def _np_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x -= x.max()
    ex = np.exp(x)
    return ex / ex.sum()


def avg_token_chr_length(tokens: list[str]) -> float:
    if isinstance(tokens, (list, tuple)):
        tokens = list(tokens)
    else:
        raise ValueError("tokens must be a list/tuple of token strings")

    total_length = sum(len(token.encode("utf-8")) for token in tokens)
    return total_length / len(tokens)


def compression_ratio_agnostic(token_ids, raw_text_utf8: str, *, vocab_size=None):
    if vocab_size is None:
        raise ValueError("vocab_size must be provided for agnostic compression ratio")

    # model side
    used = len(set(token_ids))
    model_bits = used * np.log2(vocab_size)
    model_bits /= 8.0  # convert to bytes

    # raw side
    raw_bytes = len(raw_text_utf8.encode("utf-8"))
    raw_bits = raw_bytes * 8.0
    ratio = model_bits / max(1.0, raw_bits)
    return {
        "ratio_model_over_raw": ratio,
        "compression_gain_x": (1.0 / ratio) if ratio > 0 else float("inf"),
        "model_bits": model_bits,
        "raw_bits": raw_bits,
    }


def vocab_utilization(token_ids, vocab_size):
    used = len(set(token_ids))
    return {"utilization": used / vocab_size, "unique_types": used}


def heaps_curve(token_ids, *, step=10_000, max_points=2_000_000):
    """
    Build cumulative (N, V) pairs to study Heaps' law. step controls resolution.

    Parameters:
    - token_ids: list of token IDs (e.g. from a tokenizer)
    - step: how many tokens to skip between points
    - max_points: maximum number of points to return

    Returns:
    - N_list: cumulative number of tokens
    - V_list: cumulative number of unique types (vocabulary size)
    """
    N_list, V_list = [], []
    seen = set()
    N = 0
    limit = min(len(token_ids), max_points)
    for i in range(limit):
        seen.add(token_ids[i])
        N += 1
        if N % step == 0 or N == limit:
            N_list.append(N)
            V_list.append(len(seen))
    return np.array(N_list, dtype=np.int64), np.array(V_list, dtype=np.int64)


def fit_heaps_law(N, V, use_prefix=0.8):
    """
    Fit Heaps' law to the data (N, V) using linear regression in log-log space.
    Returns K (intercept) and beta (slope) of the fitted line V = K * N^beta.

    Parameters:
    - N: array of cumulative token counts
    - V: array of cumulative unique type counts
    - use_prefix: fraction of data to use for fitting (default 0.8)

    Returns:
    - K: intercept of the fitted line
    - beta: slope of the fitted line
    - r2: coefficient of determination (R^2)
    """
    cut = int(len(N) * use_prefix)
    x = np.log(N[:cut])
    y = np.log(V[:cut])
    A = np.vstack([np.ones_like(x), x]).T
    # least squares
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    logK, beta = coef[0], coef[1]
    # R^2
    y_hat = A @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(np.exp(logK)), float(beta), float(r2)


def plot_heaps(N, V, K=None, beta=None, ax=None, title="Heaps' law"):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Linear
    ax[0].plot(N, V)
    ax[0].set_xlabel("Tokens N")
    ax[0].set_ylabel("Types V")
    ax[0].set_title(title + " (linear)")

    # Log–log
    ax[1].plot(np.log(N), np.log(V))
    ax[1].set_xlabel("log N")
    ax[1].set_ylabel("log V")
    t = title + " (log-log)"
    if K is not None and beta is not None:
        # overlay fitted line in log–log space
        x = np.log(N)
        y_fit = np.log(K) + beta * x
        ax[1].plot(x, y_fit)
        t += f"\nfit: V≈{K:.2f}·N^{beta:.3f}"
    ax[1].set_title(t)
    return ax


# region hparam search


def fit_tokenizer_params(
    *,
    train_text_path: str,
    val_text_path: str,
    merges_grid=(100, 250, 500, 1000, 2000, 5000, 10000),
    tokenizer_trainer=None,
    special_tokens: dict = TOK_SPECIAL_TOKENS,
    min_frequency: int = 2,
    lowercase: bool = False,
    add_prefix_space: bool = True,
    # N* for Heaps extrapolation; if None -> 5x len(val_ids)
    target_tokens: int | None = None,
    heaps_step: int = 200,
    verbose: bool = True,
    plot_heaps: bool = False,
    csv_log_path: str | None = None,
):
    if tokenizer_trainer is None:
        raise ValueError("pass tokenizer_trainer= train_bytelevel_bpe")
    if special_tokens is None:
        special_tokens = {"pad": "<pad>", "unk": "<unk>", "bos": "<bos>", "eos": "<eos>"}

    rows = []

    with open(train_text_path, "r", encoding="utf-8") as f:
        train_text = f.read().strip()
    with open(val_text_path, "r", encoding="utf-8") as f:
        val_text = f.read().strip()

    # tokenizer is trained ONLY on train; we encode val with it
    for m in merges_grid:
        if verbose:
            print(f"\n==> merges={m}")

        tok = tokenizer_trainer(
            files=[train_text_path],
            merges=m,
            min_frequency=min_frequency,
            lowercase=lowercase,
            add_prefix_space=add_prefix_space,
            save_filename=f"bpe_{m}",
        )

        # encode without auto special tokens
        encode = partial(tok.encode, add_special_tokens=False)
        decode = tok.decode
        pad_id = tok.token_to_id(special_tokens["bos"]) if special_tokens.get("bos") else 0  # NOTE

        train_ids = encode(train_text).ids
        val_ids = encode(val_text).ids

        Vmax = tok.get_vocab_size()

        # tokenizer efficiency
        chars_per_token = (len(val_text) / max(1, len(val_ids))) if len(val_ids) > 0 else float("inf")
        tokens_per_1k_chars = (len(val_ids) / max(1, len(val_text) / 1000.0)) if len(val_text) > 0 else float("inf")

        # heaps curve & fit (on val stream)
        N, V = heaps_curve(val_ids, step=len(val_ids) // heaps_step, max_points=len(val_ids))
        K, beta, r2 = fit_heaps_law(N, V, use_prefix=0.8)

        V_end = len(set(val_ids))
        util_now = V_end / max(1, Vmax)

        N_star = target_tokens if target_tokens is not None else int(5 * len(val_ids))
        V_hat_star = min(Vmax, (K * (N_star**beta)) if (not np.isnan(K) and not np.isnan(beta)) else np.nan)
        util_star = (V_hat_star / Vmax) if (Vmax and not np.isnan(V_hat_star)) else np.nan

        compression_ratio = compression_ratio_agnostic(val_ids, val_text, vocab_size=Vmax)

        # same for train ids
        chars_per_token_train = (len(train_text) / max(1, len(train_ids))) if len(train_ids) > 0 else float("inf")
        tokens_per_1k_chars_train = (
            (len(train_ids) / max(1, len(train_text) / 1000.0)) if len(train_text) > 0 else float("inf")
        )

        N_train, V_train = heaps_curve(train_ids, step=len(train_ids) // heaps_step, max_points=len(train_ids))
        K_train, beta_train, r2_train = fit_heaps_law(N_train, V_train, use_prefix=0.8)

        V_end_train = len(set(train_ids))
        util_now_train = V_end_train / max(1, Vmax)

        N_star_train = target_tokens if target_tokens is not None else int(5 * len(train_ids))
        V_hat_star_train = min(
            Vmax,
            (K_train * (N_star_train**beta_train)) if (not np.isnan(K_train) and not np.isnan(beta_train)) else np.nan,
        )
        util_star_train = (V_hat_star_train / Vmax) if (Vmax and not np.isnan(V_hat_star_train)) else np.nan

        compression_ratio_train = compression_ratio_agnostic(train_ids, train_text, vocab_size=Vmax)

        rows.append(
            {
                "merges": m,
                "vocab_size": Vmax,
                # val stats
                "K": K,
                "beta": beta,
                "R2": r2,
                "util_now": util_now,
                "compression_ratio": compression_ratio["ratio_model_over_raw"],
                "chars_per_token": chars_per_token,
                "tokens_per_1k_chars": tokens_per_1k_chars,
                "N_star": N_star,
                "V_end": V_end,
                "V_hat_star": V_hat_star,
                "util_star": util_star,
                # train stats
                "train_K": K_train,
                "train_beta": beta_train,
                "train_R2": r2_train,
                "train_util_now": util_now_train,
                "train_compression_ratio": compression_ratio_train["ratio_model_over_raw"],
                "train_chars_per_token": chars_per_token_train,
                "train_tokens_per_1k_chars": tokens_per_1k_chars_train,
                "train_N_star": N_star_train,
                "train_V_end": V_end_train,
                "train_V_hat_star": V_hat_star_train,
                "train_util_star": util_star_train,
            }
        )

        if plot_heaps:

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(N, V)
            ax[0].set_title(f"Heaps (linear) k={m}")
            ax[0].set_xlabel("N")
            ax[0].set_ylabel("V")
            # log–log + fit
            ax[1].plot(np.log(N), np.log(V))
            # if not (np.isnan(K) or np.isnan(beta)):
            x = np.log(N)
            y_fit = np.log(K) + beta * x
            ax[1].plot(x, y_fit)
            ax[1].set_title(f"Heaps (log-log) k={m}\nV≈{K:.1f}·N^{beta:.3f} (R²={r2:.3f})")
            plt.tight_layout()
            plt.show()

    df = pd.DataFrame(rows).sort_values("merges").reset_index(drop=True)
    if csv_log_path:
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        df.to_csv(csv_log_path, index=False)

    return df


def plot_heaps_and_metrics(
    df,
    *,
    recommended_k_range=None,
    suptitle="Heaps' & Tokenizer Search Analysis",
    plot_train_metrics=True,
    save_path=None,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axK, axB, axU, axCPT, axCOMP, axTp1k = axes.ravel()

    def _plot(ax, ycol, ylabel, title):
        (p_val,) = ax.plot(df["merges"], df[ycol], marker=".", linewidth=2, label="Validation set")
        if plot_train_metrics and f"train_{ycol}" in df.columns:
            ax.plot(df["merges"], df[f"train_{ycol}"], marker=".", linewidth=1, alpha=0.7, label="Train set")

        ax.set_title(title, pad=8)
        ax.set_xlabel("Merges (k)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # vertical guide lines (no label here → we’ll add one handle globally)
        if recommended_k_range is not None:
            ks = (
                recommended_k_range
                if isinstance(recommended_k_range, (list, tuple, np.ndarray))
                else [recommended_k_range]
            )
            for k in ks:
                ax.axvline(k, linestyle="--", linewidth=1.5, alpha=0.6, color="green")

        return p_val

    _plot(axK, "K", "Heaps K", "Heaps K vs. merges")
    _plot(axB, "beta", "Heaps β", "Heaps β vs. merges")
    _plot(axU, "util_now", "Vocab utilization", "Utilization vs. merges")
    _plot(axCPT, "chars_per_token", "Chars per token", "Chars per token vs. merges")
    _plot(axCOMP, "compression_ratio", "Compression ratio (model/raw)", "Compression ratio vs. merges")
    _plot(axTp1k, "tokens_per_1k_chars", "Tokens per 1k chars", "Tokens per 1k chars vs. merges")

    fig.suptitle(suptitle, y=1.06, fontsize=14)

    # build a single legend
    handles, labels = axK.get_legend_handles_labels()

    # add one synthetic handle for the green range lines
    if recommended_k_range is not None:
        import matplotlib.lines as mlines

        line = mlines.Line2D(
            [],
            [],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Recommended k range ({recommended_k_range[0]}-{recommended_k_range[-1]})",
        )
        handles.append(line)
        labels.append(f"Recommended k range ({recommended_k_range[0]}-{recommended_k_range[-1]})")

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, axes
