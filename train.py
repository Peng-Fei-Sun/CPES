
import os
from args import get_args
args = get_args()
if args["deterministic_and_reproducible"]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"



import torch

import torch.nn.functional as F
import math
import random
import time
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataset import GraphAnmDataset, collate_fn
from model import Model

from scipy.stats import spearmanr



if args["deterministic_and_reproducible"]:

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)




def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    os.environ["PYTHONHASHSEED"] = str(seed)



def move_to_device(batch_tuple, device):
    return tuple(x.to(device) for x in batch_tuple)


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_target_mean_std(dataset, eps: float = 1e-8):
    n = 0
    s1 = 0.0
    s2 = 0.0
    for i in range(len(dataset)):
        sample = dataset[i]
        y = sample["data_complex"].y.view(-1).to(dtype=torch.float32, device="cpu")
        n += int(y.numel())
        s1 += float(y.sum().item())
        s2 += float((y * y).sum().item())
    mean = s1 / n
    var = max(0.0, (s2 / n) - (mean * mean))
    std = max(float(math.sqrt(var)), float(eps))
    return float(mean), float(std)


def train_one_epoch(model, loader, optimizer, device, max_grad_norm, target_mean=None, target_std=None):
    model.train()
    total_sse_raw = 0.0
    num_batches = 0
    num_clipped = 0
    for batch in loader:
        num_batches += 1
        batch = move_to_device(batch, device)
        batch_complex = batch[0]
        y_raw = batch_complex.y.view(-1).to(dtype=torch.float32)
        if target_mean is not None and target_std is not None:
            y = (y_raw - float(target_mean)) / float(target_std)
        else:
            y = y_raw

        pred = model(batch)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        grad_norm_val = float(grad_norm)
        if math.isfinite(grad_norm_val) and grad_norm_val > max_grad_norm:
            num_clipped += 1
        optimizer.step()

        if target_mean is not None and target_std is not None:
            pred_raw = pred * float(target_std) + float(target_mean)
        else:
            pred_raw = pred
        total_sse_raw += F.mse_loss(pred_raw, y_raw, reduction="sum").item()
    train_rmse = math.sqrt(total_sse_raw / len(loader.dataset))
    clip_ratio = (num_clipped / num_batches) if num_batches > 0 else 0.0
    return train_rmse, clip_ratio


@torch.no_grad()
def evaluate(model, loader, device, target_mean=None, target_std=None):
    model.eval()
    total_sse_raw = 0.0
    preds = []
    targets = []
    for batch in loader:
        batch = move_to_device(batch, device)
        batch_complex = batch[0]
        y_raw = batch_complex.y.view(-1).to(dtype=torch.float32)
        if target_mean is not None and target_std is not None:
            y = (y_raw - float(target_mean)) / float(target_std)
        else:
            y = y_raw

        pred = model(batch)
        if target_mean is not None and target_std is not None:
            pred_raw = pred * float(target_std) + float(target_mean)
        else:
            pred_raw = pred

        total_sse_raw += F.mse_loss(pred_raw, y_raw, reduction="sum").item()
        preds.append(pred_raw.detach().cpu())
        targets.append(y_raw.detach().cpu())

    rmse = math.sqrt(total_sse_raw / len(loader.dataset))

    pred_all = torch.cat(preds, dim=0).numpy()
    y_all = torch.cat(targets, dim=0).numpy()
    pearson = np.corrcoef(pred_all, y_all)[0, 1] if pred_all.size > 1 else float("nan")



    mae = np.mean(np.abs(pred_all - y_all))


    A = np.vstack([pred_all, np.ones_like(pred_all)]).T
    b, a = np.linalg.lstsq(A, y_all, rcond=None)[0]
    y_fit = a + b * pred_all
    residual = y_all - y_fit
    sd = np.sqrt(np.sum(residual ** 2) / (len(y_all) - 1))


    spearman = spearmanr(y_all, pred_all)[0]

    return rmse, float(pearson), float(mae), float(sd), float(spearman)


def train_once(run_args):
    data_root = run_args["data_root"]
    batch_size = run_args["batch_size"]
    epochs = run_args["epochs"]
    lr = run_args["lr"]
    seed = run_args["seed"]
    patience = run_args["patience"]
    optimizer_name = run_args["optimizer"]
    weight_decay = run_args["weight_decay"]
    lr_patience = run_args["lr_patience"]
    lr_factor = run_args["lr_factor"]
    min_lr = run_args["min_lr"]
    K_modes_ligand = run_args["K_modes_ligand"]
    K_u_ligand = run_args["K_u_ligand"]
    K_modes_protein = run_args["K_modes_protein"]
    K_u_protein = run_args["K_u_protein"]
    K_modes_complex = run_args["K_modes_complex"]
    K_u_complex = run_args["K_u_complex"]
    use_warmup = run_args["use_warmup"]
    warmup_epochs = run_args["warmup_epochs"]
    warmup_start_factor = run_args["warmup_start_factor"]
    max_grad_norm = run_args["max_grad_norm"]
    run_id = run_args["run_id"]
    normalize_targets = bool(run_args.get("normalize_targets", False))
    target_norm_eps = float(run_args.get("target_norm_eps", 1e-8))

    set_seed(seed)

    train_set = GraphAnmDataset(os.path.join(data_root, "train"))
    val_set = GraphAnmDataset(os.path.join(data_root, "valid"))
    test2013_set = GraphAnmDataset(os.path.join(data_root, "test2013"))
    test2016_set = GraphAnmDataset(os.path.join(data_root, "test2016"))
    test2019_set = GraphAnmDataset(os.path.join(data_root, "test2019"))
    testcsar_set = GraphAnmDataset(os.path.join(data_root, "testcsar"))




    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False, persistent_workers=False)
    test2013_loader = DataLoader(test2013_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False, persistent_workers=False)
    test2016_loader = DataLoader(test2016_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False, persistent_workers=False)
    test2019_loader = DataLoader(test2019_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False, persistent_workers=False)
    testcsar_loader = DataLoader(testcsar_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False, persistent_workers=False)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(K_modes_ligand=K_modes_ligand, K_u_ligand=K_u_ligand,
                  K_modes_protein=K_modes_protein, K_u_protein=K_u_protein,
                  K_modes_complex=K_modes_complex, K_u_complex=K_u_complex,
                  node_dim=35, hidden_dim=256).to(device)
    print(f"[Model] trainable parameters: {count_trainable_params(model):,}")

    target_mean = None
    target_std = None
    if normalize_targets:
        target_mean, target_std = compute_target_mean_std(train_set, eps=target_norm_eps)
        print(f"[TargetNorm] Z-score enabled | mean={target_mean:.6f} std={target_std:.6f}")

    optim_name = optimizer_name.lower()
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'adam' or 'adamw'.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )
    if use_warmup:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: warmup_start_factor + (1.0 - warmup_start_factor) * min(1.0, (e + 1) / warmup_epochs),
        )

    start_epoch = 1
    best_rmse = float("inf")
    epochs_no_improve = 0
    results_dir = os.path.join(".", "results")
    os.makedirs(results_dir, exist_ok=True)
    best_ckpt_path = os.path.join(results_dir, f"best_model_run{run_id:03d}.pt")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_rmse, clip_ratio = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_grad_norm,
            target_mean=target_mean,
            target_std=target_std,
        )
        rmse, pearson, _, _, _ = evaluate(model, val_loader, device, target_mean=target_mean, target_std=target_std)
        if use_warmup and epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(rmse)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        if rmse < best_rmse:
            best_rmse = rmse
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "target_mean": target_mean, "target_std": target_std},
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:04d} | train_RMSE {train_rmse:.4f} | val_RMSE {rmse:.4f} | "
            f"val_Pearson {pearson:.4f} | lr {current_lr:.2e} | "
            f"clip {clip_ratio:.2%} | no_improve {epochs_no_improve:03d} | runtime {elapsed:.2f}s"
        )
        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch:03d}")
            break

    print(f"Best val RMSE: {best_rmse:.6f}")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if normalize_targets:
            target_mean = float(ckpt.get("target_mean", target_mean))
            target_std = float(ckpt.get("target_std", target_std))

    rmse_test2013, pearson_test2013, _, _, _ = evaluate(model, test2013_loader, device, target_mean=target_mean, target_std=target_std)
    rmse_test2016, pearson_test2016, _, _, _ = evaluate(model, test2016_loader, device, target_mean=target_mean, target_std=target_std)
    rmse_test2019, pearson_test2019, _, _, _ = evaluate(model, test2019_loader, device, target_mean=target_mean, target_std=target_std)
    rmse_testcsar, pearson_testcsar, _, _, _ = evaluate(model, testcsar_loader, device, target_mean=target_mean, target_std=target_std)



    print(f"Test2013 RMSE: {rmse_test2013:.6f} | Pearson: {pearson_test2013:.6f}")
    print(f"Test2016 RMSE: {rmse_test2016:.6f} | Pearson: {pearson_test2016:.6f}")
    print(f"Test2019 RMSE: {rmse_test2019:.6f} | Pearson: {pearson_test2019:.6f}")
    print(f"TestCSAR RMSE: {rmse_testcsar:.6f} | Pearson: {pearson_testcsar:.6f}")


    return {
        "seed": seed,
        "val_rmse": best_rmse,
        "test2013_rmse": rmse_test2013,
        "test2013_pr": pearson_test2013,
        "test2016_rmse": rmse_test2016,
        "test2016_pr": pearson_test2016,
        "test2019_rmse": rmse_test2019,
        "test2019_pr": pearson_test2019,
        "testcsar_rmse": rmse_testcsar,
        "testcsar_pr": pearson_testcsar,
    }


if __name__ == "__main__":
    
    args = get_args()
    
    num_runs = args["num_runs"]
    stop_file = "stop.txt"
    results_all = []
    seed_gen = random.Random()
    seed_always = args["seed_always"]
    used = set(seed_always)
    for run_idx in range(10**10):
        run_args = args.copy()
        if run_idx < len(seed_always):
            seed = seed_always[run_idx]
            tag = "fixed"
        else:

            seed = seed_gen.randint(0, 2 ** 31 - 1)
            while seed in used:

                seed = seed_gen.randint(0, 2 ** 31 - 1)
            used.add(seed)
            tag = "random"
        run_args["seed"] = seed
        run_args["run_id"] = run_idx + 1
        print(f"\n===== Run {run_idx + 1:03d} ({tag}) | seed {seed} =====")
        results_all.append(train_once(run_args))







        if os.path.exists(stop_file):
            with open(stop_file, 'r') as f:
                stop_target = int(f.read().strip())
        else:
            stop_target = num_runs
        if  (run_idx + 1) >= stop_target:
            if stop_target != num_runs:
                print(f"\nstop.txt specifies stopping at run {stop_target}")
            break









