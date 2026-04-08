#!/usr/bin/env python3
"""
classification.py

Benchmark binary classification (BSCR vs. Control) using pretrained CNN
architectures: ResNet50, VGG16, EfficientNet-B0, DenseNet121.

Features:
  - Patient-level train / validation / test split (no data leakage)
  - Class-weighted CrossEntropyLoss to handle cohort imbalance
  - Backbone freezing warmup then full fine-tuning
  - ReduceLROnPlateau scheduler
  - Early stopping on validation loss
  - Mixed-precision training (AMP)
  - Best-model checkpointing

Usage:
    python classification.py \
        --metadata metadata_final.xlsx \
        --images /path/to/images \
        --epochs 15 \
        --batch-size 64 \
        --models resnet50 vgg16 efficientnet_b0 densenet121
"""

import argparse, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ---- Dataset ----
class FundusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(int(r["label"]), dtype=torch.long)

TRAIN_TFM = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
    transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
EVAL_TFM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

# ---- Model factory ----
def get_model(name, nc=2):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, nc); hp = "fc."
    elif name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, nc); hp = "classifier.6"
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc); hp = "classifier.1"
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, nc); hp = "classifier"
    else: raise ValueError(name)
    bb = [n for n, _ in m.named_parameters() if not n.startswith(hp)]
    return m, bb

def set_backbone(model, bb_names, freeze):
    p = dict(model.named_parameters())
    for n in bb_names:
        if n in p: p[n].requires_grad = not freeze

# ---- Evaluate ----
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); crit = nn.CrossEntropyLoss()
    labels, preds, probs, loss_sum, nb = [], [], [], 0., 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            o = model(x); loss_sum += crit(o, y).item(); nb += 1
        labels.extend(y.cpu().numpy()); preds.extend(o.argmax(1).cpu().numpy())
        probs.extend(torch.softmax(o,1)[:,1].cpu().numpy())
    auc = roc_auc_score(labels, probs) if len(set(labels))>1 else float("nan")
    return {"AUC-ROC": auc, "Accuracy": accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds, zero_division=0),
            "Recall": recall_score(labels, preds, zero_division=0),
            "F1-Score": f1_score(labels, preds, zero_division=0)}, loss_sum/max(nb,1)

# ---- Train ----
def train_model(model, bb, train_ld, val_ld, device, cw, args):
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    if args.warmup_epochs > 0:
        set_backbone(model, bb, True); print(f"  Backbone frozen for {args.warmup_epochs} epochs")
    best_vl, best_ep, wait = float("inf"), 0, 0
    ckpt = f"_ckpt_{id(model)}.pt"; t0 = time.time()
    for ep in range(1, args.epochs+1):
        if ep == args.warmup_epochs+1:
            set_backbone(model, bb, False); print(f"  Backbone unfrozen at epoch {ep}")
        model.train(); rl = 0.
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); rl += loss.item()
        vm, vl = evaluate(model, val_ld, device); sched.step(vl)
        print(f"  Epoch {ep}/{args.epochs}  train_loss={rl/len(train_ld):.4f}  "
              f"val_loss={vl:.4f}  val_AUC={vm['AUC-ROC']:.4f}  val_F1={vm['F1-Score']:.4f}")
        if vl < best_vl - 1e-4:
            best_vl, best_ep, wait = vl, ep, 0; torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= args.patience: print(f"  Early stopping at epoch {ep}"); break
    print(f"  Done in {(time.time()-t0)/60:.1f} min. Best epoch: {best_ep}")
    if os.path.exists(ckpt): model.load_state_dict(torch.load(ckpt, map_location=device)); os.remove(ckpt)
    return model

# ---- Main ----
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True; print(f"Device: {device}")
    df = pd.read_excel(args.metadata)
    df["image_path"] = df.apply(lambda r: os.path.join(args.images, r["cohort"], r["filename"]), axis=1)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["label"] = (df["cohort"]=="BCR").astype(int); print(f"Valid images: {len(df)}")

    # Patient-level 3-way split
    g1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tr_full_i, te_i = next(g1.split(df, groups=df["patient_id"]))
    tr_full, test_df = df.iloc[tr_full_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)
    g2 = GroupShuffleSplit(1, test_size=0.10/0.80, random_state=SEED)
    tr_i, va_i = next(g2.split(tr_full, groups=tr_full["patient_id"]))
    train_df, val_df = tr_full.iloc[tr_i].reset_index(drop=True), tr_full.iloc[va_i].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    cw = compute_class_weight("balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
    cw = torch.tensor(cw, dtype=torch.float32).to(device); print(f"Class weights: {cw}")

    lkw = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_ld = DataLoader(FundusDataset(train_df, TRAIN_TFM), shuffle=True, drop_last=True, **lkw)
    val_ld = DataLoader(FundusDataset(val_df, EVAL_TFM), shuffle=False, **lkw)
    test_ld = DataLoader(FundusDataset(test_df, EVAL_TFM), shuffle=False, **lkw)

    results = []
    for name in args.models:
        print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
        model, bb = get_model(name); model = model.to(device)
        model = train_model(model, bb, train_ld, val_ld, device, cw, args)
        m, tl = evaluate(model, test_ld, device); m["Model"] = name; results.append(m)
        print(f"  TEST -> AUC={m['AUC-ROC']:.4f}  Acc={m['Accuracy']:.4f}  F1={m['F1-Score']:.4f}")

    rdf = pd.DataFrame(results).set_index("Model")
    print(f"\n{'='*60}\n  RESULTS\n{'='*60}\n{rdf.to_string()}")
    out = os.path.join(args.images, "benchmark_results.csv"); rdf.to_csv(out); print(f"Saved: {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BSCR vs Control CNN benchmark")
    p.add_argument("--metadata", required=True); p.add_argument("--images", required=True)
    p.add_argument("--models", nargs="+", default=["resnet50","vgg16","efficientnet_b0","densenet121"])
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=5, help="Frozen backbone warmup epochs")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    main(p.parse_args())
