"""
CIFAR-10 Training — ResNet-20 or Vision Transformer
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
from models.resnet20 import ResNet20
from models.vit import ViTCIFAR10
from optimizers.muon import SingleDeviceMuonWithAuxAdam


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(data_dir, batch_size, num_workers=4, val_fraction=0.1,
                    augment=False):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.2023, 0.1994, 0.2010),
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_full = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                     download=True, transform=train_transform)
    test_set   = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                     download=True, transform=test_transform)

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
# Model factory
# ---------------------------------------------------------------------------

def build_model(args):
    """Instantiate the model from CLI arguments."""
    m = args.model.lower()
    if m == "resnet20":
        return ResNet20(num_classes=10)
    elif m == "vit":
        return ViTCIFAR10(
            img_size=32,
            patch_size=4,
            num_classes=10,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
        )
    else:
        raise ValueError(f"Unknown model: {m}. Choose from: resnet20, vit")


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
        # For ViT: exclude embedding layers, norms, biases, and cls/pos tokens
        # For ResNet: exclude BN layers and biases
        def is_muon_param(name, p):
            if p.ndim < 2:
                return False
            # Always exclude these regardless of model
            exclude_keywords = ["bias", "bn", "norm",        # norms & biases
                                 "cls_token", "pos_embed",    # ViT special tokens
                                 "patch_embed.proj"]                    
            return not any(kw in name for kw in exclude_keywords)

        muon_params = [p for name, p in model.named_parameters()
                       if is_muon_param(name, p)]
        adam_params = [p for name, p in model.named_parameters()
                       if not is_muon_param(name, p)]

        param_groups = [
            dict(params=muon_params, lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay, use_muon=True),
            dict(params=adam_params, lr=args.lr_adamw,
                betas=(args.beta1, args.beta2), eps=args.eps,
                weight_decay=args.weight_decay, use_muon=False),
        ]

        # Temporary Debug Check inside build_optimizer()
        print("\n--- OPTIMIZER PARAMETER ASSIGNMENT VERIFICATION ---")
        print("Going to MUON:")
        for name, p in model.named_parameters():
            if is_muon_param(name, p):
                print(f" -> {name} | Shape: {list(p.shape)}")

        print("\nGoing to ADAMW:")
        for name, p in model.named_parameters():
            if not is_muon_param(name, p):
                print(f" -> {name} | Shape: {list(p.shape)}")
        print("-" * 50 + "\n")
        
        return SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {opt}. "
                         "Choose from: sgd, adam, adamw, rmsprop, adagrad, sam, muon")


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
        milestones = args.lr_milestones or [100, 150]
        return torch.optim.lr_scheduler.MultiStepLR(
            base_opt, milestones=milestones, gamma=args.lr_gamma)
    elif sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=args.epochs, eta_min=args.min_lr)
    elif sched == "cosine_warmup":
        # Linear warmup + cosine decay — strongly recommended for ViT
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(base_opt, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {sched}. "
                         "Choose from: none, step, cosine, cosine_warmup")


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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5

            optimizer.first_step(zero_grad=True)
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
# Metrics helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        description="CIFAR-10 training — ResNet-20 or ViT, optimizer minima investigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Experiment ---
    p.add_argument("--run_name",   type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./runs")
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--seed",       type=int, default=42)

    # --- Model ---
    p.add_argument("--model", type=str, default="resnet20",
                   choices=["resnet20", "vit"],
                   help="Model architecture")


    # --- Training ---
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--augment",     action="store_true",
                   help="Enable RandAugment (strongly recommended for ViT)")

    # --- Optimizer ---
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw", "rmsprop", "adagrad", "sam", "muon"])
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--nesterov",     action="store_true")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps",   type=float, default=1e-8)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--lr_adamw",     type=float, default=3e-4)
    p.add_argument("--rho",          type=float, default=0.05)
    p.add_argument("--adaptive_sam", action="store_true")
    p.add_argument("--base_optimizer", type=str, default="sgd",
                   choices=["sgd", "adam", "adamw"])

    # --- Scheduler ---
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "step", "cosine", "cosine_warmup"])
    p.add_argument("--warmup_epochs",  type=int,   default=10,
                   help="Linear warmup epochs (used with cosine_warmup)")
    p.add_argument("--lr_milestones",  type=int,   nargs="+", default=None)
    p.add_argument("--lr_gamma",       type=float, default=0.1)
    p.add_argument("--min_lr",         type=float, default=0.0)

    # --- Checkpointing ---
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume",     type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_name is None:
        extras = ""
        if args.optimizer == "sam":
            extras = f"_rho{args.rho}_base{args.base_optimizer}"
        elif args.optimizer == "sgd":
            extras = f"_mom{args.momentum}" + ("_nesterov" if args.nesterov else "")
        elif args.model == "vit":
            extras = ""
        args.run_name = (
            f"{args.model}_{args.optimizer}{extras}"
            f"_lr{args.lr}_wd{args.weight_decay}"
            f"_bs{args.batch_size}_{args.scheduler}_seed{args.seed}"
        )

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb.init(
        project="OptiML_Minima",
        name=args.run_name,
        config=vars(args),
    )

    device = torch.device(
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Run    : {args.run_name}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers,
        augment=args.augment,
    )

    model = build_model(args).to(device)
    print(f"{args.model} — {count_parameters(model):,} trainable parameters\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))

    start_epoch = 0
    best_acc    = 0.0
    best_loss   = float("inf")
    history     = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        best_loss   = ckpt.get("best_loss", float("inf"))
        history     = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch} (best acc = {best_acc:.2f}%)\n")

    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device)

        if scheduler is not None:
            scheduler.step()

        val_loss,  val_acc  = evaluate(model, val_loader,  criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = (
            optimizer.base_optimizer.param_groups[0]["lr"]
            if isinstance(optimizer, SAM)
            else optimizer.param_groups[0]["lr"]
        )

        record = {
            "epoch":       epoch + 1,
            "lr":          current_lr,
            "train_loss":  round(train_loss,  6),
            "train_acc":   round(train_acc,   4),
            "val_loss":    round(val_loss,    6),
            "val_acc":     round(val_acc,     4),
            "test_loss":   round(test_loss,   6),
            "test_acc":    round(test_acc,    4),
            "time_s":      round(elapsed,     2),
            "grad_norm":   round(grad_norm,   6),
            "weight_norm": round(compute_weight_norm(model), 6),
        }
        history.append(record)
        wandb.log(record)

        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"LR {current_lr:.2e} | "
            f"Train loss {train_loss:.4f}  acc {train_acc:5.2f}% | "
            f"Val   loss {val_loss:.4f}  acc {val_acc:5.2f}% | "
            f"Test  loss {test_loss:.4f}  acc {test_acc:5.2f}% | "
            f"{elapsed:.1f}s"
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc  = val_acc
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc":  best_acc,
                "best_loss": best_loss,
                "history":   history,
                "args":      vars(args),
            }, out_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc":  best_acc,
                "history":   history,
                "args":      vars(args),
            }, out_dir / f"epoch_{epoch+1:03d}.pt")

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
