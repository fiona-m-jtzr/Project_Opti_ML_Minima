"""
CIFAR-10 Training with ResNet-20
Investigating the loss landscape / minima shape of different optimizers.
"""

import argparse
import time
import json
import math
from pathlib import Path
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from optimizers.sam import SAM
from optimizers.lbfgs import LBFGS
from models.resnet20 import ResNet20
from optimizers.muon import SingleDeviceMuonWithAuxAdam


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(data_dir, batch_size, num_workers=4, val_fraction=0.1):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2023, 0.1994, 0.2010),
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_full = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                     download=True, transform=transform)
    test_set   = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                     download=True, transform=transform)

    val_size   = int(len(train_full) * val_fraction)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(model, args):
    """Instantiate the optimizer from CLI arguments."""
    opt = args.optimizer.lower()
    params = model.parameters()

    if opt == "sgd":
        return torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=args.nesterov,
        )
    elif opt == "adam":
        return torch.optim.Adam(
            params, lr=args.lr,
            betas=(args.beta1, args.beta2), eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif opt == "adamw":
        return torch.optim.AdamW(
            params, lr=args.lr,
            betas=(args.beta1, args.beta2), eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif opt == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=args.lr, alpha=args.alpha,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )
    elif opt == "adagrad":
        return torch.optim.Adagrad(
            params, lr=args.lr, weight_decay=args.weight_decay,
        )
    elif opt == "sam":
        base_cls = {
            "sgd":   torch.optim.SGD,
            "adam":  torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }.get(args.base_optimizer.lower())
        if base_cls is None:
            raise ValueError(f"Unsupported SAM base optimizer: {args.base_optimizer}")
        base_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
        if args.base_optimizer.lower() == "sgd":
            base_kwargs.update(momentum=args.momentum, nesterov=args.nesterov)
        return SAM(params, base_cls, rho=args.rho,
                   adaptive=args.adaptive_sam, **base_kwargs
        )
    elif opt == "muon":
        muon_params = [
            p for name, p in model.named_parameters()
            if p.ndim >= 2 and "bn" not in name and "bias" not in name
        ]
        adam_params = [
            p for name, p in model.named_parameters()
            if p.ndim < 2 or "bn" in name or "bias" in name
        ]
        param_groups = [
            dict(params=muon_params,
                lr=0.02,
                momentum=0.95,
                weight_decay=0.0, 
                use_muon=True),
            dict(params=adam_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=0.0,
                use_muon=False),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {opt}. "
                         "Choose from: sgd, adam, adamw, rmsprop, adagrad, sam")


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, args, steps_per_epoch):
    """Build a learning-rate scheduler."""
    sched = args.scheduler.lower()
    base_opt = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer

    if sched == "none":
        return None
    elif sched == "step":
        # Decay by gamma at milestones (default: 100 and 150 for 200-epoch run)
        milestones = args.lr_milestones or [100, 150]
        return torch.optim.lr_scheduler.MultiStepLR(
            base_opt, milestones=milestones, gamma=args.lr_gamma)
    elif sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=args.epochs, eta_min=args.min_lr)
    elif sched == "warmup_cosine":
        # Linear warm-up then cosine decay
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps  = args.epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(base_opt, lr_lambda)
    elif sched == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(
            base_opt, base_lr=args.min_lr, max_lr=args.lr,
            step_size_up=steps_per_epoch * 5, mode="triangular2",
            cycle_momentum=False,
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched}. "
                         "Choose from: none, step, cosine, warmup_cosine, cyclic")


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    is_sam = isinstance(optimizer, SAM)
    grad_norm_sum = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if is_sam:
            # Step 1: forward + backward at w
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5

            optimizer.first_step(zero_grad=True)

            # Step 2: forward + backward at w + e(w)
            outputs = model(inputs)
            criterion(outputs, targets).backward()
            optimizer.second_step(zero_grad=True)

        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5

            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
        grad_norm_sum += grad_norm

    return total_loss / total, 100.0 * correct / total, grad_norm_sum / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Metrics helpers  (for loss-landscape analysis)
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Weight norm
# ---------------------------------------------------------------------------

def compute_weight_norm(model):
    return sum(
        p.norm().item() ** 2
        for p in model.parameters()
        if p.requires_grad
    ) ** 0.5

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-20 training — optimizer minima investigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment ---
    p.add_argument("--run_name",   type=str, default=None,
                   help="Tag for this run (auto-generated from settings if None)")
    p.add_argument("--output_dir", type=str, default="./runs",
                   help="Directory for checkpoints and logs")
    p.add_argument("--data_dir",   type=str, default="./data",
                   help="CIFAR-10 download/cache directory")
    p.add_argument("--seed",       type=int, default=42,
                   help="Global random seed")

    # --- Training ---
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label-smoothing factor for cross-entropy (0 = off)")
    p.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience in epochs")

    # --- Optimizer ---
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw", "rmsprop", "adagrad", "sam", "muon"],
                   help="Optimizer to use")
    p.add_argument("--lr",           type=float, default=0.1,  help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay")
    # SGD / SAM-SGD
    p.add_argument("--momentum",  type=float, default=0.9,  help="SGD momentum")
    p.add_argument("--nesterov",  action="store_true",      help="Nesterov momentum")
    # Adam / AdamW
    p.add_argument("--beta1", type=float, default=0.9,   help="Adam beta1")
    p.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    p.add_argument("--eps",   type=float, default=1e-8,  help="Adam epsilon")
    # RMSprop
    p.add_argument("--alpha", type=float, default=0.99, help="RMSprop smoothing constant")
    # LBFGS specific
    p.add_argument("--max_iter",        type=int,   default=20,
                help="Max iterations per step for L-BFGS")
    p.add_argument("--history_size",    type=int,   default=100,
                help="L-BFGS history size")
    p.add_argument("--line_search_fn",  type=str,   default="strong_wolfe",
                choices=["strong_wolfe", "none"],
                help="Line search for L-BFGS (strong_wolfe strongly recommended)")
    # SAM specific
    p.add_argument("--rho",           type=float, default=0.05, help="SAM neighbourhood size")
    p.add_argument("--adaptive_sam",  action="store_true",      help="Use adaptive SAM (ASAM)")
    p.add_argument("--base_optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw"],
                   help="Base optimizer for SAM")

    # --- Scheduler ---
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "step", "cosine", "warmup_cosine", "cyclic"],
                   help="LR scheduler")
    p.add_argument("--lr_milestones", type=int, nargs="+", default=None,
                   help="Epochs for MultiStepLR decay (scheduler=step)")
    p.add_argument("--lr_gamma",     type=float, default=0.1,
                   help="Decay factor for MultiStepLR")
    p.add_argument("--min_lr",       type=float, default=0.0,
                   help="Minimum LR for cosine/cyclic schedulers")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Warm-up epochs for warmup_cosine scheduler")

    # --- Checkpointing ---
    p.add_argument("--save_every",   type=int, default=10,
                   help="Save a checkpoint every N epochs (0 = only best)")
    p.add_argument("--resume",       type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Auto-generate run name
    if args.run_name is None:
        extras = ""
        if args.optimizer == "sam":
            extras = f"_rho{args.rho}_base{args.base_optimizer}"
        elif args.optimizer == "sgd":
            extras = f"_mom{args.momentum}" + ("_nesterov" if args.nesterov else "")
        args.run_name = (
            f"{args.optimizer}{extras}_lr{args.lr}_wd{args.weight_decay}"
            f"_bs{args.batch_size}_{args.scheduler}_seed{args.seed}"
        )

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    wandb.init(
        project="OptiML_Minima",
        name=args.run_name,
        config=vars(args),
    )

    # Device
    device = torch.device(
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Run : {args.run_name}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers,
    )

    # Model
    model = ResNet20(num_classes=10).to(device)
    print(f"ResNet-20 — {count_parameters(model):,} trainable parameters\n")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    
    # Optionally resume
    start_epoch = 0
    best_acc    = 0.0
    best_loss = float("inf")
    history     = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        best_loss = ckpt.get("best_loss", float("inf"))
        history     = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch} (best acc = {best_acc:.2f}%)\n")
    
    patience = args.patience
    epochs_no_improve = 0

    # ---------- Training loop ----------
    step_scheduler_per_batch = args.scheduler in ("warmup_cosine", "cyclic")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device)

        # Step schedulers that track epochs
        if scheduler and not step_scheduler_per_batch:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = (
            optimizer.base_optimizer.param_groups[0]["lr"]
            if isinstance(optimizer, SAM)
            else optimizer.param_groups[0]["lr"]
        )

        record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_loss":  round(val_loss,  6),
            "val_acc":   round(val_acc,   4),
            "test_loss":  round(test_loss,  6),
            "test_acc":   round(test_acc,   4),
            "time_s":     round(elapsed,    2),
            "grad_norm": round(grad_norm, 6),
            "weight_norm": round(compute_weight_norm(model), 6),
        }
        history.append(record)
        wandb.log(record) 

        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"LR {current_lr:.2e} | "
            f"Train loss {train_loss:.4f}  acc {train_acc:5.2f}% | "
            f"Validation loss {val_loss:.4f}  acc {val_acc:5.2f}% | "
            f"Test loss {test_loss:.4f}  acc {test_acc:5.2f}% | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_loss = val_loss 
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_loss": best_loss,
                "history": history,
                "args": vars(args),
            }, out_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

        # Periodic checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "history": history,
                "args": vars(args),
            }, out_dir / f"epoch_{epoch+1:03d}.pt")

        # Flush history to disk (useful for live monitoring)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n✓ Training complete. Best val accuracy: {best_acc:.2f}%")
    print(f"  Outputs saved to: {out_dir}\n")
    artifact = wandb.Artifact(
        name=f"model-{args.run_name}",
        type="model",
        metadata={"best_val_acc": best_acc, "best_val_loss": best_loss},
    )
    artifact.add_file(str(out_dir / "best.pt"))
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()