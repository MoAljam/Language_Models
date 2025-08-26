import os, json, time, uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, List, Tuple
from itertools import product

from torch.optim import AdamW
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from train import train, ConfigTrain
from neural_bigram import NeuralBigram, ConfigNeuralBigram
from GPT import GPT, ConfigGPT
from utils import WarmupThenCosine, init_dataloader, count_params
from bpe_hf import train_and_encode_tokenizer, train_bytelevel_bpe, TOK_SPECIAL_TOKENS, TOK_SAVE_DIR


@dataclass
class HparamsSpace:
    """
    abstract base class for hparam search spaces
    all fields must be non-empty iterables
    """

    def __post_init__(self):
        # all showuld be iterables
        assert all(
            hasattr(v, "__iter__") or hasattr(v, "__getitem__") for k, v in self.__dict__.items()
        ), "All hparam space values must be iterables"
        # no empty iterables
        assert all(len(v) > 0 for k, v in self.__dict__.items()), "All hparam space iterables must be non-empty"

    def num_total_combinations(self) -> int:
        return np.prod([len(v) for k, v in self.__dict__.items()])


@dataclass
class HparamsSpaceNBigram(HparamsSpace):
    merges: Iterable[int] = (200,)
    dropout: Iterable[float] = (0.2,)
    lr: Iterable[float] = (3e-3,)
    lr_scheduler: Iterable[str] = ("cosine_warmup",)  # ("cosine", "cosine_restarts", cosine_warmup, ...)


@dataclass
class HparamsSpaceGPT(HparamsSpace):
    merges: Iterable[int] = (200,)
    n_embed: Iterable[int] = (64,)
    n_heads: Iterable[int] = (4,)
    n_layers: Iterable[int] = (4,)
    dropout: Iterable[float] = (0.2,)
    lr: Iterable[float] = (3e-3,)
    lr_scheduler: Iterable[str] = ("cosine_warmup",)


def _trial_run_name(base: str = "sweep") -> str:
    return f"{base}_{time.strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _safe_lr_sched(name: str, optimizer, total_steps: int, eta_min: float):
    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    if name == "cosine_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=max(2, total_steps // 10), T_mult=2, eta_min=eta_min)
    if name == "cosine_warmup":
        return WarmupThenCosine(optimizer, warmup_steps=500, T_max=total_steps, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler: {name}")


# region nbigram hparam search
def hparams_search_nBigram(
    *,
    # search space
    hp_space: HparamsSpaceNBigram,
    # training
    base_cfg_train: ConfigTrain,
    base_cfg_model: ConfigNeuralBigram,
    train_text_path: str,
    val_text_path: str,
    # tokenizer
    tokenizer_trainer: callable = train_bytelevel_bpe,  # train_bytelevel_bpe
    special_tokens: dict = TOK_SPECIAL_TOKENS,
    tok_min_frequency: int = 2,
    tok_save_dir: str = TOK_SAVE_DIR,
    # data
    batch_size: int = 32,
    block_size: int = 128,
    # lr scheduler and optimizer
    eta_min: float = 1e-8,
    weight_decay: float = 1e-4,
    verbose: bool = False,
):
    os.makedirs(base_cfg_train.log_dir, exist_ok=True)
    os.makedirs(base_cfg_train.ckpt_dir, exist_ok=True)
    os.makedirs(tok_save_dir, exist_ok=True)

    total_runs = hp_space.num_total_combinations()

    rows: List[Dict[str, Any]] = []
    pbar_all = tqdm(total=total_runs, desc="HParams Search", leave=True, unit="trial")

    for merges in hp_space.merges:
        # create unique tokenizer name per merges
        tok_name = f"bpe_{merges}m_" + uuid.uuid4().hex[:8] + ".json"
        tok_info = train_and_encode_tokenizer(
            tokenizer_trainer=tokenizer_trainer,
            train_text_path=train_text_path,
            other_texts_paths={"val": val_text_path},
            merges=merges,
            min_frequency=tok_min_frequency,
            special_tokens=special_tokens,
            save_dir=tok_save_dir,
            save_filename=tok_name,
        )
        train_ids = tok_info["train_ids"]
        val_ids = tok_info["other_texts_ids"]["val"]
        vocab_sz = tok_info["vocab_size"]

        # dataloaders
        train_loader = init_dataloader(train_ids, block_size, batch_size, train=True, shuffle=True)
        val_loader = init_dataloader(val_ids, block_size, batch_size, train=False, shuffle=True)

        for lr, dropout, sched_name in product(hp_space.lr, hp_space.dropout, hp_space.lr_scheduler):

            run_name = _trial_run_name("hps")
            run_dir = os.path.join(base_cfg_train.log_dir, run_name)
            writer = SummaryWriter(log_dir=run_dir, flush_secs=5)
            # update model and train cfgs
            cfg_model = type(base_cfg_model)(
                **{
                    **vars(base_cfg_model),
                    "vocab_size": vocab_sz,
                    "dropout": dropout,
                }
            )
            cfg_train = type(base_cfg_train)(
                **{
                    **vars(base_cfg_train),
                    "ckpt_best_filename": f"best_{run_name}.pt",
                    "ckpt_last_filename": f"last_{run_name}.pt",
                    "log_dir": run_dir,
                }
            )

            # build model
            model = NeuralBigram(cfg_model)
            model.to(cfg_train.device)
            # get model size
            model_size = count_params(model)
            # compile if possible
            try:
                if cfg_train.device == "cpu":
                    model.compile(mode="reduce-overhead")
                else:
                    model.compile()
            except Exception as e:
                print(f"Warning: model.compile() failed with error: {e}. Continuing without compilation.")
            # optimizer
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # scheduler
            total_steps = cfg_train.epochs * max(1, len(train_loader) // max(1, cfg_train.grad_accum_steps))
            lr_scheduler = _safe_lr_sched(sched_name, optimizer, total_steps, eta_min)
            # scaler
            scaler = GradScaler(enabled=cfg_train.use_amp)

            # train
            out = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_train,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=scaler,
                writer=writer,
                show_pbar=False,
            )

            # log all hparams
            rl_scheduler_state_dict = (
                lr_scheduler.state_dict()
                if lr_scheduler.__class__.__name__ != "WarmupThenCosine"
                else {k: v for k, v in lr_scheduler.state_dict().items() if k != "cosine"}
            )
            hparams_json = {
                "tokenizer": {
                    "type": "bytelevel_bpe",
                    "n_merges": merges,
                    "min_frequency": tok_min_frequency,
                    "special_tokens": special_tokens,
                    "path": os.path.join(tok_save_dir, tok_name),
                },
                "data": {
                    "batch_size": batch_size,
                    "block_size": block_size,
                },
                "model": {"cfg": vars(cfg_model), "model_size": model_size},
                "optimizer": {
                    "type": optimizer.__class__.__name__,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "defaults": optimizer.defaults,
                },
                "lr_scheduler": {
                    "type": lr_scheduler.__class__.__name__,
                    "eta_min": eta_min,
                    "state_dict": rl_scheduler_state_dict,
                },
                "scaler": {
                    "type": scaler.__class__.__name__,
                    "enabled": scaler.is_enabled(),
                    "state_dict": scaler.state_dict(),
                },
                "training": {"cfg": vars(cfg_train).copy()},
            }

            full_config_file = os.path.join(cfg_train.ckpt_dir, f"{run_name}_hparams.json")
            with open(full_config_file, "w") as f:
                json.dump(hparams_json, f, indent=4)

            writer.add_text("config/json", "```json\n" + json.dumps(hparams_json, indent=2) + "\n```", 0)
            writer.flush()
            writer.close()

            # final metrics per run
            row = {
                "merges": merges,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "scheduler": sched_name,
                "vocab_size": vocab_sz,
                "model_size": model_size,
                "val_ppl": out["best_val_ppl"],
                "train_ppl": out["history"]["train_ppl"][-1],
                "val_loss": out["history"]["val_loss"][-1],
                "train_loss": out["history"]["train_loss"][-1],
                "epochs": out["last_epoch"] + 1,
                "ckpt_best": os.path.join(cfg_train.ckpt_best_path),
                "ckpt_last": os.path.join(cfg_train.ckpt_last_path),
                "full_config_file": full_config_file,
                "run": run_name,
            }
            rows.append(row)

            if verbose:
                print(
                    f"[{run_name}] k={merges}, lr={lr:.1e}, drop={dropout}, "
                    f"sched={sched_name} -> val_ppl {row['val_ppl']:.2f}"
                )

            pbar_all.set_postfix(
                val_ppl=f"{row['val_ppl']:.2f}",
                merges=merges,
                lr=f"{lr:.1e}",
                dropout=dropout,
                weight_decay=weight_decay,
                scheduler=sched_name,
            )
            pbar_all.update(1)

    pbar_all.close()
    df = pd.DataFrame(rows).sort_values(["val_ppl", "merges", "lr", "dropout"]).reset_index(drop=True)
    return df


# region GPT hparam search


def hparams_search_GPT(
    *,
    # search space
    hp_space: HparamsSpaceGPT,
    # training
    base_cfg_train: ConfigTrain,
    base_cfg_model: ConfigGPT,
    train_text_path: str,
    val_text_path: str,
    # tokenizer
    tokenizer_trainer: callable = train_bytelevel_bpe,  # train_bytelevel_bpe
    special_tokens: dict = TOK_SPECIAL_TOKENS,
    tok_min_frequency: int = 2,
    tok_save_dir: str = TOK_SAVE_DIR,
    # data
    batch_size: int = 32,
    block_size: int = 128,
    # lr scheduler and optimizer
    eta_min: float = 1e-8,
    weight_decay: float = 1e-4,
    verbose: bool = False,
):
    os.makedirs(base_cfg_train.log_dir, exist_ok=True)
    os.makedirs(base_cfg_train.ckpt_dir, exist_ok=True)
    os.makedirs(tok_save_dir, exist_ok=True)

    total_runs = hp_space.num_total_combinations()

    rows: List[Dict[str, Any]] = []
    pbar_all = tqdm(total=total_runs, desc="HParams Search", leave=True, unit="trial")

    for merges in hp_space.merges:
        # create unique tokenizer name per merges
        tok_name = f"bpe_{merges}m_" + uuid.uuid4().hex[:8] + ".json"
        tok_info = train_and_encode_tokenizer(
            tokenizer_trainer=tokenizer_trainer,
            train_text_path=train_text_path,
            other_texts_paths={"val": val_text_path},
            merges=merges,
            min_frequency=tok_min_frequency,
            special_tokens=special_tokens,
            save_dir=tok_save_dir,
            save_filename=tok_name,
        )
        train_ids = tok_info["train_ids"]
        val_ids = tok_info["other_texts_ids"]["val"]
        vocab_sz = tok_info["vocab_size"]

        # dataloaders
        train_loader = init_dataloader(train_ids, block_size, batch_size, train=True, shuffle=True)
        val_loader = init_dataloader(val_ids, block_size, batch_size, train=False, shuffle=True)

        for n_embed, n_heads, n_layers, dropout, lr, sched_name in product(
            hp_space.n_embed,
            hp_space.n_heads,
            hp_space.n_layers,
            hp_space.dropout,
            hp_space.lr,
            hp_space.lr_scheduler,
        ):

            run_name = _trial_run_name("hps")
            run_dir = os.path.join(base_cfg_train.log_dir, run_name)
            writer = SummaryWriter(log_dir=run_dir, flush_secs=5)
            # update model and train cfgs
            cfg_model = type(base_cfg_model)(
                **{
                    **vars(base_cfg_model),
                    "vocab_size": vocab_sz,
                    "n_embed": n_embed,
                    "n_head": n_heads,
                    "n_layer": n_layers,
                    "dropout": dropout,
                }
            )
            cfg_train = type(base_cfg_train)(
                **{
                    **vars(base_cfg_train),
                    "ckpt_best_filename": f"best_{run_name}.pt",
                    "ckpt_last_filename": f"last_{run_name}.pt",
                    "log_dir": run_dir,
                }
            )
            # build model
            model = GPT(cfg_model)
            model.to(cfg_train.device)
            # get model size
            model_size = count_params(model)
            # compile if possible
            try:
                if cfg_train.device == "cpu":
                    model.compile(mode="reduce-overhead")
                else:
                    model.compile()
            except Exception as e:
                print(f"Warning: model.compile() failed with error: {e}. Continuing without compilation.")

            # optimizer
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # scheduler
            total_steps = cfg_train.epochs * max(1, len(train_loader) // max(1, cfg_train.grad_accum_steps))
            lr_scheduler = _safe_lr_sched(sched_name, optimizer, total_steps, eta_min)
            # scaler
            scaler = GradScaler(enabled=cfg_train.use_amp)

            # train
            out = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_train,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=scaler,
                writer=writer,
                show_pbar=False,
            )

            # log all hparams
            rl_scheduler_state_dict = (
                lr_scheduler.state_dict()
                if lr_scheduler.__class__.__name__ != "WarmupThenCosine"
                else {k: v for k, v in lr_scheduler.state_dict().items() if k != "cosine"}
            )
            hparams_json = {
                "tokenizer": {
                    "type": "bytelevel_bpe",
                    "n_merges": merges,
                    "min_frequency": tok_min_frequency,
                    "special_tokens": special_tokens,
                    "path": os.path.join(tok_save_dir, tok_name),
                },
                "data": {
                    "batch_size": batch_size,
                    "block_size": block_size,
                },
                "model": {"cfg": vars(cfg_model).copy(), "model_size": model_size},
                "optimizer": {
                    "type": optimizer.__class__.__name__,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "defaults": optimizer.defaults,
                },
                "lr_scheduler": {
                    "type": lr_scheduler.__class__.__name__,
                    "eta_min": eta_min,
                    "state_dict": rl_scheduler_state_dict,
                },
                "scaler": {
                    "type": scaler.__class__.__name__,
                    "enabled": scaler.is_enabled(),
                    "state_dict": scaler.state_dict(),
                },
                "training": {"cfg": vars(cfg_train).copy()},
            }

            full_config_file = os.path.join(cfg_train.ckpt_dir, f"{run_name}_hparams.json")
            with open(full_config_file, "w") as f:
                json.dump(hparams_json, f, indent=4)

            writer.add_text("config/json", "```json\n" + json.dumps(hparams_json, indent=2) + "\n```", 0)
            writer.flush()
            writer.close()

            # final metrics per run
            row = {
                "merges": merges,
                "n_embed": n_embed,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "scheduler": sched_name,
                "vocab_size": vocab_sz,
                "model_size": model_size,
                "val_ppl": out["best_val_ppl"],
                "train_ppl": out["history"]["train_ppl"][-1],
                "val_loss": out["history"]["val_loss"][-1],
                "train_loss": out["history"]["train_loss"][-1],
                "epochs": out["last_epoch"] + 1,
                "ckpt_best": os.path.join(cfg_train.ckpt_best_path),
                "ckpt_last": os.path.join(cfg_train.ckpt_last_path),
                "full_config_file": full_config_file,
                "run": run_name,
            }
            rows.append(row)

            if verbose:
                print(
                    f"[{run_name}] k={merges}, lr={lr:.1e}, drop={dropout}, "
                    f"sched={sched_name} -> val_ppl {row['val_ppl']:.2f}"
                )

            pbar_all.set_postfix(
                val_ppl=f"{row['val_ppl']:.2f}",
                merges=merges,
                lr=f"{lr:.1e}",
                dropout=dropout,
                weight_decay=weight_decay,
                scheduler=sched_name,
            )
            pbar_all.update(1)

    pbar_all.close()
    df = (
        pd.DataFrame(rows)
        .sort_values(["val_ppl", "merges", "n_embed", "n_heads", "n_layers", "lr", "dropout"])
        .reset_index(drop=True)
    )
    return df


# region analysis / ploting


def _ci95(std: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = np.maximum(1, n.astype(float))
    return 1.96 * (std / np.sqrt(n))


def _maybe_ax(ax: Optional[plt.Axes]) -> Tuple[plt.Axes, bool]:
    """Return (ax, created_new_flag)."""
    if ax is None:
        _, ax = plt.subplots()
        return ax, True
    return ax, False


# (grouped) per-HP effect (mean +- 95% CI)
def plot_per_hp_grouped(
    df: pd.DataFrame,
    hp: str,
    *,
    groupby: Optional[str] = None,
    metric: str = "val_ppl",
    max_groups: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    plot mean(metric) ± 95% CI vs hp.
    - If `groupby` is None: a single line/errorbar plot (no grouping).
    - If `groupby` is provided: one line/errorbar per group value.
    returns the axes used.
    """
    # required columns
    required_cols = {hp, metric} if groupby is None else {hp, metric, groupby}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.copy()
    ax, created = _maybe_ax(ax)

    # no grouping
    if groupby is None:
        # aggregate over hp only
        tab = df.groupby(hp)[metric].agg(mean="mean", std="std", count="count").reset_index().sort_values(hp)
        tab["ci95"] = _ci95(tab["std"].values, tab["count"].values)

        x_vals = tab[hp].to_numpy()
        x_pos = np.arange(len(x_vals))
        means = tab["mean"].to_numpy()
        errs = tab["ci95"].to_numpy()

        ax.errorbar(x_pos, means, yerr=errs, fmt="o-", capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_vals, rotation=45)
        ax.set_xlabel(hp)
        ax.set_ylabel(f"mean {metric} ± 95% CI")
        ax.set_title(f"{hp} effect")
        ax.grid(True, linestyle="-", color="gray", linewidth=0.7, alpha=0.5)
        if created:
            plt.tight_layout()
        return ax

    # grouped
    tab = (
        df.groupby([groupby, hp])[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values([groupby, hp])
    )
    tab["ci95"] = _ci95(tab["std"].values, tab["count"].values)
    # if more than max_groups unique values, plot heatmap instead
    n_groups = len(tab[groupby].unique())
    if n_groups > max_groups:
        print(f"Warning: {n_groups} unique values for {groupby}, defaulting to heatmap instead with no CI.")
        return plot_interaction_heatmap(
            tab, hp, groupby, metric="mean", title=f"{metric} interaction: {hp} vs {groupby}", ax=ax
        )

    x_vals = np.sort(tab[hp].unique())
    x_pos = np.arange(len(x_vals))
    idx_map = {xv: i for i, xv in enumerate(x_vals)}

    for gval in np.sort(tab[groupby].unique()):
        sub = tab[tab[groupby] == gval]
        means = np.full_like(x_pos, np.nan, dtype=float)
        errs = np.full_like(x_pos, np.nan, dtype=float)
        for _, r in sub.iterrows():
            i = idx_map[r[hp]]
            means[i] = r["mean"]
            errs[i] = r["ci95"]
        mask = ~np.isnan(means)
        ax.errorbar(x_pos[mask], means[mask], yerr=errs[mask], fmt="o-", capsize=3, label=f"{groupby}={gval}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_vals, rotation=45)
    ax.set_xlabel(hp)
    ax.set_ylabel(f"mean {metric} ± 95% CI over {hp}")
    ax.set_title(f"{hp} effect by {groupby}")
    ax.legend()
    ax.grid(True, linestyle="-", color="gray", linewidth=0.7, alpha=0.5)

    if created:
        plt.tight_layout()
    return ax


# global effect via RF + permutation importance
def rf_perm_importance(
    df: pd.DataFrame,
    features: List[str],
    *,
    metric: str = "val_ppl",
    n_estimators: int = 500,
    n_repeats: int = 30,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    fits RF on target and computes permutation importance.
    returns a df with feature, mean, std, ci95 (on the means).
    """
    df = df.copy()
    X = df[features].copy()
    y = df[metric].values

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    ).fit(X, y)

    pi = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    mean = pi.importances_mean
    std = pi.importances_std
    # 95% CI for the mean decrease (std of mean = std/sqrt(n_repeats))
    ci = 1.96 * std / np.sqrt(n_repeats)

    out = (
        pd.DataFrame(
            {
                "feature": features,
                "perm_mean": mean,
                "perm_std": std,
                "perm_ci95": ci,
            }
        )
        .sort_values("perm_mean", ascending=False)
        .reset_index(drop=True)
    )
    return out


def plot_perm_importance_hbar(
    df: pd.DataFrame,
    features: List[str],
    *,
    ax: Optional[plt.Axes] = None,
    metric: str = "val_ppl",
    title: str = "Permutation importance on val_ppl",
    **kwargs,
) -> plt.Axes:
    """
    Horizontal bar plot with 95% CI whiskers and a zero line.
    Expects columns: feature, perm_mean, perm_ci95
    """
    ax, created = _maybe_ax(ax)
    dfp = rf_perm_importance(df, features, metric=metric, **kwargs)
    y = np.arange(len(dfp))
    ax.barh(y, dfp["perm_mean"].values, xerr=dfp["perm_ci95"].values, capsize=4)
    ax.axvline(0.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(dfp["feature"].values)
    ax.set_xlabel("Permutation importance (Δ error on log(val_ppl))")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.invert_yaxis()  # largest at top
    # vertical grid lines
    ax.grid(which="major", axis="x", linestyle="--", color="gray", linewidth=0.7, alpha=0.5)

    if created:
        plt.tight_layout()
    return ax


# interaction heatmaps
def plot_interaction_heatmap(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    metric: str = "val_ppl",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    heatmap of mean(metric) over a grid of (xcol, ycol).
    """
    df = df.copy()
    if title is None:
        title = f"{metric} interaction: {xcol} vs {ycol}"

    agg = df.groupby([xcol, ycol])[metric].mean().reset_index()

    xs = np.sort(agg[xcol].unique())
    ys = np.sort(agg[ycol].unique())
    Xi = {v: i for i, v in enumerate(xs)}
    Yi = {v: j for j, v in enumerate(ys)}

    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for _, r in agg.iterrows():
        Z[Yi[r[ycol]], Xi[r[xcol]]] = r[metric]

    ax, created = _maybe_ax(ax)
    im = ax.imshow(Z, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=45)
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)
    if created:
        plt.tight_layout()
    return ax


# Generalization gap vs model_size
def plot_generelization_gap(
    df: pd.DataFrame,
    *,
    hp="model_size",
    groupby: Optional[str] = None,
    use_log: bool = True,
    max_groups: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    line plot: mean ± 95% CI of (val - train) on ppl or log-ppl vs hp model_size.
    """
    required_cols = {"train_ppl", "val_ppl", hp}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.dropna(subset=["train_ppl", "val_ppl"]).copy()
    if use_log:
        df["gen_gap"] = np.log(df["val_ppl"]) - np.log(df["train_ppl"])
        ylabel = "generalization gap (log val_ppl - log train_ppl)"
    else:
        df["gen_gap"] = df["val_ppl"] - df["train_ppl"]
        ylabel = "generalization gap (val_ppl - train_ppl)"

    # no grouping
    if groupby is None:
        tab = df.groupby(hp)["gen_gap"].agg(mean="mean", std="std", count="count").sort_index()
        ci = _ci95(tab["std"].values, tab["count"].values)

        ax, created = _maybe_ax(ax)
        ax.errorbar(tab.index.values, tab["mean"].values, yerr=ci, fmt="o-", capsize=3)
        ax.set_xlabel(hp)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Generalization gap vs {hp} (mean ± 95% CI)")
        if created:
            plt.tight_layout()
        return ax

    # grouped
    tab = (
        df.groupby([groupby, hp])["gen_gap"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values([groupby, hp])
    )
    tab["ci95"] = _ci95(tab["std"].values, tab["count"].values)

    # if more than max_groups unique groupby values, plot heatmap instead
    if len(tab[groupby].unique()) > max_groups:
        print(f"Warning: {groupby} has more than 5 unique values. Plotting heatmap instead.")
        return plot_interaction_heatmap(
            tab, xcol=hp, ycol=groupby, metric="mean", title=f"Generalization gap interaction: {hp} vs {groupby}", ax=ax
        )

    x_vals = np.sort(tab[hp].unique())
    x_pos = np.arange(len(x_vals))
    idx_map = {xv: i for i, xv in enumerate(x_vals)}
    ax, created = _maybe_ax(ax)
    for gval in np.sort(tab[groupby].unique()):
        sub = tab[tab[groupby] == gval]
        means = np.full_like(x_pos, np.nan, dtype=float)
        errs = np.full_like(x_pos, np.nan, dtype=float)
        for _, r in sub.iterrows():
            i = idx_map[r[hp]]
            means[i] = r["mean"]
            errs[i] = r["ci95"]
        mask = ~np.isnan(means)
        ax.errorbar(x_pos[mask], means[mask], yerr=errs[mask], fmt="o-", capsize=3, label=f"{groupby}={gval}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_vals, rotation=45)
    ax.set_xlabel(hp)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Generalization gap vs {hp} by {groupby} (mean ± 95% CI)")
    ax.legend()
    ax.grid(True, linestyle="-", color="gray", linewidth=0.7, alpha=0.5)

    if created:
        plt.tight_layout()
    return ax
