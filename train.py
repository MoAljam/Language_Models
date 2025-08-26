import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from typing import Optional, Dict, Any
import os
from dataclasses import dataclass
from utils import set_seed, save_checkpoint, load_checkpoint


@dataclass
class ConfigTrain:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    use_amp: bool = True
    grad_accum_steps: int = 1  #
    # max_grad_norm: float = 1.0
    early_stop_patience: int = 5
    early_stop_tolerance: float = 0  # tolerance for early stopping
    seed: int = 666
    eval_interval: int = 5
    ckpt_interval: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt_best_filename: str = "best.pt"
    ckpt_last_filename: str = "last.pt"
    log_dir: str = "logs"
    ckpt_best_path: str = None
    ckpt_last_path: str = None

    def __post_init__(self):
        # ensure checkpoint directory exists
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.ckpt_last_path = os.path.join(self.ckpt_dir, self.ckpt_last_filename)
        self.ckpt_best_path = os.path.join(self.ckpt_dir, self.ckpt_best_filename)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on the given DataLoader and return the loss (mean negative log-likelihood (NLL)).
    """
    model.eval()
    total_nll = 0.0
    total_tok = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        B, T, V = logits.shape
        # CE on logits here is equavilant to NLL
        nll_sum = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="sum")

        total_nll += nll_sum.item()
        total_tok += B * T

    mean_nll = total_nll / max(1, total_tok)
    # for perplexity, exponentiate the mean NLL
    return mean_nll


@torch.no_grad()
def evaluate_ppl(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on the given DataLoader and return the perplexity.
    """
    model.eval()
    mean_nll = evaluate(model, loader, device)
    return torch.exp(torch.tensor(mean_nll)).item()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ConfigTrain,
    resume_from: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    writer: Optional[SummaryWriter] = None,
    verbose: bool = False,
    show_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Train the model with the given training and validation DataLoaders and configuration.
    Supports resuming from a checkpoint.

    Parameters:
    - model: The neural network model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - cfg: Training configuration (TrainConfig).
    - resume_from: Optional path to a checkpoint to resume training from.
    - optimizer: Optional optimizer. If None, AdamW with default params is used.
    - scheduler: Optional learning rate scheduler. If None, CosineAnnealingLR is used.
    - scaler: Optional GradScaler for mixed precision training. If None, a new one is created.
    - writer: Optional SummaryWriter for TensorBoard logging.
    - verbose: If True, prints progress messages.
    - show_pbar: If True, shows progress bars.

    Returns:
    - history: A dictionary containing training and validation loss and perplexity history.
    """
    set_seed(cfg.seed)
    device = cfg.device
    if device is None:
        device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_steps = cfg.epochs * max(1, len(train_loader) // max(1, cfg.grad_accum_steps))

    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    if scaler is None:
        scaler = GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    best_val_loss = torch.inf
    best_train_loss = torch.inf
    no_improve = 0
    earrly_stop = False

    val_avg_loss = torch.inf
    val_ppl = torch.inf
    epoch = 0

    log_each_n_step = cfg.eval_interval * len(train_loader) // 4
    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        no_improve = ckpt.get("no_improve", 0)
        if verbose:
            print(f"Resumed from {resume_from} (epoch {start_epoch})")

    #### training loop ####
    history = {"train_loss": [], "train_ppl": [], "val_ppl": [], "val_loss": [], "epoch": []}
    pbar_epoch = tqdm(
        range(start_epoch, cfg.epochs), total=cfg.epochs, desc="Training Progress", leave=True, disable=not show_pbar
    )
    for epoch in pbar_epoch:
        model.train(True)
        running_loss = 0.0
        running_loss_sum = 0.0
        running_tokens = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{cfg.epochs}",
            leave=False,
            disable=not show_pbar,
        )
        for step, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=cfg.device, enabled=cfg.use_amp):
                logits = model(x)
                B, T, V = logits.shape
                # for comparative nll log, sum CE loss
                loss_sum = F.cross_entropy(
                    logits.view(B * T, V),
                    y.view(B * T),
                    reduction="sum",
                )
            loss = loss_sum / (B * T)  # mean loss
            # scale loss for gradient accumulation
            loss_to_backprop = loss / cfg.grad_accum_steps
            scaler.scale(loss_to_backprop).backward()
            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item()
            running_loss_sum += loss_sum.item()
            running_tokens += B * T

            # train_avg_loss = running_loss / max(1, step + 1)
            train_avg_loss = running_loss_sum / max(1, running_tokens)

            # write model graph to TensorBoard
            if writer is not None:
                global_step = epoch * len(train_loader) + step
                if global_step % log_each_n_step == 0:
                    writer.add_scalar("train/loss_step_mean", loss.item(), global_step)
                    writer.add_scalar("train/loss_step_sum", loss_sum.item(), global_step)

        # model.train(False)
        train_ppl = torch.exp(torch.tensor(train_avg_loss)).item()

        # each eval_interval epochs, run evaluation
        if (epoch + 1) % cfg.eval_interval == 0 or epoch == cfg.epochs - 1:
            #### model evaluation ####
            val_avg_loss = evaluate(model, val_loader, device)
            # train_avg_loss = evaluate(model, train_loader, device)
            val_ppl = torch.exp(torch.tensor(val_avg_loss)).item()

            # early stopping update on validation ppl
            if val_avg_loss < best_val_loss - cfg.early_stop_tolerance:
                no_improve = 0
            else:
                no_improve += 1
                earrly_stop = cfg.early_stop_patience and no_improve >= cfg.early_stop_patience
                if verbose:
                    print(f"No improvement ({no_improve}/{cfg.early_stop_patience}).")

            #### logging ####
            history["train_loss"].append(train_avg_loss)
            history["train_ppl"].append(train_ppl)
            history["val_loss"].append(val_avg_loss)
            history["val_ppl"].append(val_ppl)
            history["epoch"].append(epoch)

            if writer is not None:
                writer.add_scalars(
                    "epoch/loss",
                    {
                        "train_loss": train_avg_loss,
                        "val_loss": val_avg_loss,
                    },
                    epoch,
                )
                writer.add_scalars(
                    "epoch/ppl",
                    {
                        "train_ppl": train_ppl,
                        "val_ppl": val_ppl,
                    },
                    epoch,
                )

        # on non-eval epochs, check early stopping on train ppl
        else:
            # early stopping update on training ppl
            if train_avg_loss < best_train_loss - cfg.early_stop_tolerance:
                no_improve = 0
            else:
                no_improve += 1
                earrly_stop = cfg.early_stop_patience and no_improve >= cfg.early_stop_patience
                if verbose:
                    print(f"No improvement ({no_improve}/{cfg.early_stop_patience}).")

        # checkpoint
        if (epoch + 1) % cfg.ckpt_interval == 0 or epoch == cfg.epochs - 1 or earrly_stop:
            ckpt_path = cfg.ckpt_last_path
            if val_avg_loss < best_val_loss:
                ckpt_path = cfg.ckpt_best_path

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "no_improve": no_improve,
                    "loss": val_avg_loss,
                    "config": vars(cfg),
                },
                ckpt_path,
            )

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss  # update best loss
        if train_avg_loss < best_train_loss:
            best_train_loss = train_avg_loss

        # update epoch progress bar
        pbar_epoch.set_postfix(
            train_loss=f"{train_avg_loss:.4f}",
            train_ppl=f"{train_ppl:.2f}",
            val_loss=f"{val_avg_loss:.4f}",
            val_ppl=f"{val_ppl:.2f}",
            no_improve=f"{no_improve}/{cfg.early_stop_patience}" if cfg.early_stop_patience else "N/A",
        )

        # early stopping check
        if earrly_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}.")
            break

    if writer is not None:
        writer.flush()
        writer.close()

    return {
        "history": history,
        "best_val_ppl": torch.exp(torch.tensor(best_val_loss)).item(),
        "best_train_ppl": torch.exp(torch.tensor(train_avg_loss)).item(),
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "last_epoch": epoch,
    }


# not used
# @torch.no_grad()
# def compute_perplexity_explicit(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
#     # get logits, convert to probabilities with softmax, and compute perplexity explicitly
#     model.eval()
#     total_nll = 0.0  # summed CE over all tokens
#     total_tok = 0

#     for x, y in loader:
#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)
#         logits = model(x, logits=True)
#         B, T, V = logits.shape
#         probs = F.softmax(logits, dim=-1)  # (B, T, V)
#         # compute NLL
#         nll_sum = -torch.sum(torch.log(probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)), dim=-1).sum()
#         total_nll += nll_sum.item()
#         total_tok += B * T

#     mean_nll = total_nll / max(1, total_tok)  # per-token nats
#     return float(torch.exp(torch.tensor(mean_nll)))  # perplexity
