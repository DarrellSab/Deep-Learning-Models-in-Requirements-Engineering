

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error

import json
import time
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
from contextlib import nullcontext
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer


# ============================
# 0) CONFIG
# ============================
SEEDS = [13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024]

SQUAD_TRAIN_LIMIT = 60000

MAX_LEN_EMB = 128
MAX_LEN_EVAL = 384

BATCH_SIZE_EMB = 64
BATCH_SIZE_TRAIN = 64

NUM_WORKERS = 0
PIN_MEMORY = True

LR = 2e-5

EMBED_DIR = "embeddings_cache"
REGENERATE_EMBEDDINGS = False

REQ_TEST_JSON = "requirements-test-120.json"
RESULTS_CSV = "results_10runs_bert_lstm_2_layer.csv"

USE_STRICT_DETERMINISM = True

# ---- Early stopping ----
EPOCHS_MAX = 20
PATIENCE = 2
MIN_DELTA = 0.0


# ============================
# 1) Device + AMP context
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

amp_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
#scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ============================
# 2) Seeding utilities
# ============================
def set_seed(seed: int, strict: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker_init(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================
# 3) Tokenizer + frozen BERT encoder
# ============================
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
bert_model = BertModel.from_pretrained("bert-base-cased").to(device)
bert_model.eval()
for p in bert_model.parameters():
    p.requires_grad = False

print("BERT device:", next(bert_model.parameters()).device)


# ============================
# 4) Load and filter SQuAD v2 (answerable only)
# ============================
def load_filtered_squad(limit: int) -> List[dict]:
    ds = load_dataset("squad_v2")
    train_data = ds["train"].select(range(limit))

    filtered = [
        ex for ex in train_data
        if len(ex["answers"]["text"]) > 0 and len(ex["answers"]["answer_start"]) > 0
    ]
    print(f"SQuAD records used: {len(filtered)} (from {limit})")
    return filtered


# ============================
# 5) Embedding generation (ONCE) -> disk cache
# ============================
def generate_embeddings_stream(
    data: List[dict],
    tokenizer: BertTokenizerFast,
    model: BertModel,
    out_dir: str,
    max_len: int = 128,
    batch_size: int = 64
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = 0
    for start in tqdm(range(0, len(data), batch_size), desc="Generating embeddings"):
        batch = data[start:start + batch_size]
        questions = [ex["question"] for ex in batch]
        contexts = [ex["context"] for ex in batch]
        answers = [ex["answers"]["text"][0] for ex in batch]
        starts = [ex["answers"]["answer_start"][0] for ex in batch]

        enc = tokenizer(
            questions, contexts,
            max_length=max_len,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        ).to(device)

        offsets = enc.pop("offset_mapping").tolist()
        seq_ids_list = [enc.sequence_ids(i) for i in range(len(batch))]

        with torch.no_grad():
            with amp_ctx:
                emb = model(**enc).last_hidden_state.detach().cpu().numpy().astype(np.float32)

        for i in range(len(batch)):
            offs = offsets[i]
            seq_ids = seq_ids_list[i]
            s_char = starts[i]
            e_char = s_char + len(answers[i])

            s_tok = 0
            e_tok = 0

            # context tokens only (sequence_id == 1)
            for t, (off, sid) in enumerate(zip(offs, seq_ids)):
                if sid != 1:
                    continue
                if off[0] <= s_char < off[1]:
                    s_tok = t
                if off[0] < e_char <= off[1]:
                    e_tok = t

            np.save(out / f"X_{idx}.npy", emb[i])
            np.save(out / f"y_start_{idx}.npy", np.array([s_tok], dtype=np.int64))
            np.save(out / f"y_end_{idx}.npy", np.array([e_tok], dtype=np.int64))
            idx += 1

    print(f"Saved {idx} embeddings -> {out_dir}")


def ensure_embedding_cache():
    out = Path(EMBED_DIR)
    if REGENERATE_EMBEDDINGS and out.exists():
        shutil.rmtree(out)

    has_cache = out.exists() and any(p.name.startswith("X_") for p in out.iterdir())
    if has_cache:
        print(f"Embedding cache found: {EMBED_DIR} (reusing)")
        return

    print(f"Embedding cache not found. Building: {EMBED_DIR}")
    data = load_filtered_squad(SQUAD_TRAIN_LIMIT)
    generate_embeddings_stream(
        data=data,
        tokenizer=tokenizer,
        model=bert_model,
        out_dir=EMBED_DIR,
        max_len=MAX_LEN_EMB,
        batch_size=BATCH_SIZE_EMB
    )


# ============================
# 6) Disk dataset
# ============================
class QADiskDataset(Dataset):
    def __init__(self, d: str):
        self.d = d
        files = os.listdir(d)
        self.ids = sorted(int(f[2:f.index(".")]) for f in files if f.startswith("X_"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        idx = self.ids[i]
        x = np.load(f"{self.d}/X_{idx}.npy")             # [T, 768]
        y1 = np.load(f"{self.d}/y_start_{idx}.npy")[0]  # scalar
        y2 = np.load(f"{self.d}/y_end_{idx}.npy")[0]    # scalar
        return torch.from_numpy(x), torch.tensor(y1, dtype=torch.long), torch.tensor(y2, dtype=torch.long)


# ============================
# 7) LSTM QA Model (TRUE ARCHITECTURE as requested) 2 LAYER
# ============================

class QALSTMModel(nn.Module):
    def __init__(self, input_size=768, lstm1_size=512, lstm2_size=512, dense_size=512):
        super(QALSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, lstm1_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)

        
        self.lstm2 = nn.LSTM(lstm1_size, lstm2_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)

        self.projection = nn.Linear(lstm2_size, dense_size)
        self.start_out = nn.Linear(dense_size, 1)
        self.end_out = nn.Linear(dense_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = F.relu(self.projection(x))
        start_logits = self.start_out(x).squeeze(-1)
        end_logits = self.end_out(x).squeeze(-1)
        return start_logits, end_logits


# ============================
# 8) Metrics: EM / token-F1 / ROUGE-L
# ============================
def normalize_text(t: str) -> str:
    return " ".join(t.lower().split())


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_text(pred).split()
    gt = normalize_text(gold).split()
    if len(pt) == 0 and len(gt) == 0:
        return 1.0
    if len(pt) == 0 or len(gt) == 0:
        return 0.0

    pc = Counter(pt)
    gc = Counter(gt)
    common = pc & gc
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    return float(scorer.score(gold, pred)["rougeL"].fmeasure)


# ============================
# 9) RE evaluation (span extraction using frozen BERT + trained LSTM head)
# ============================
@torch.no_grad()
def get_lstm_answer(q: str, ctx: str, qa_model: QALSTMModel) -> str:
    enc = tokenizer(
        q, ctx,
        max_length=MAX_LEN_EVAL,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    ).to(device)

    offs = enc.pop("offset_mapping")[0]   # [T, 2]
    seq_ids = enc.sequence_ids(0)

    with amp_ctx:
        h = bert_model(**enc).last_hidden_state  # [1, T, 768]
        s_log, e_log = qa_model(h)               # [1, T], [1, T]

    s = int(torch.argmax(s_log, dim=-1).item())
    e = int(torch.argmax(e_log, dim=-1).item())

    if s > e:
        return ""
    if seq_ids[s] != 1 or seq_ids[e] != 1:
        return ""

    s0 = int(offs[s][0])
    e1 = int(offs[e][1])
    if s0 < 0 or e1 < 0 or e1 <= s0:
        return ""
    return ctx[s0:e1].strip()


def evaluate_requirements_json(path: str, qa_model: QALSTMModel) -> Tuple[float, float, float]:
    data = json.load(open(path, "r", encoding="utf-8"))

    em_sum = 0.0
    f1_sum = 0.0
    r_sum = 0.0

    for ex in data:
        pred = get_lstm_answer(ex["question"], ex["context"], qa_model)
        gold = ex["expected_answer"]

        em_sum += float(normalize_text(pred) == normalize_text(gold))
        f1_sum += token_f1(pred, gold)
        r_sum += rouge_l(pred, gold)

    n = len(data)
    return em_sum / n, f1_sum / n, r_sum / n


# ============================
# 10) Train/Val epoch loops (loss + strict span accuracy)
# ============================
loss_fn = nn.CrossEntropyLoss()


def train_one_epoch(model: QALSTMModel, loader: DataLoader, optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y1, y2 in loader:
        X = X.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            s_log, e_log = model(X)
            loss = loss_fn(s_log, y1) + loss_fn(e_log, y2)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

        pred_s = torch.argmax(s_log, dim=1)
        pred_e = torch.argmax(e_log, dim=1)
        correct += ((pred_s == y1) & (pred_e == y2)).sum().item()
        total += X.size(0)

    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def validate(model: QALSTMModel, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y1, y2 in loader:
        X = X.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)

        with amp_ctx:
            s_log, e_log = model(X)
            loss = loss_fn(s_log, y1) + loss_fn(e_log, y2)

        total_loss += float(loss.item())

        pred_s = torch.argmax(s_log, dim=1)
        pred_e = torch.argmax(e_log, dim=1)
        correct += ((pred_s == y1) & (pred_e == y2)).sum().item()
        total += X.size(0)

    return total_loss / max(1, len(loader)), correct / max(1, total)


# ============================
# 11) One seed run (with Early Stopping + verbose logging)
# ============================
def train_one_seed(seed: int) -> Dict[str, float]:
    set_seed(seed, strict=USE_STRICT_DETERMINISM)
    g = make_torch_generator(seed)

    ds = QADiskDataset(EMBED_DIR)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker_init,
        generator=g,
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker_init,
        generator=g,
        persistent_workers=(NUM_WORKERS > 0)
    )

    model = QALSTMModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = f"best_bert_lstm_seed_{seed}.pt"

    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_acc = validate(model, val_loader)

        print(
            f"[seed={seed}] epoch {epoch+1}/{EPOCHS_MAX} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"train_span_acc={tr_acc:.4f} val_span_acc={va_acc:.4f}"
        )

        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"[seed={seed}] New best checkpoint saved at epoch {best_epoch} (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(
                    f"[seed={seed}] Early stopping at epoch {epoch+1} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val:.4f})"
                )
                break

    train_minutes = (time.time() - t0) / 60.0

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    em, f1, rL = evaluate_requirements_json(REQ_TEST_JSON, model)

    return {
        "seed": seed,
        "epochs_max": EPOCHS_MAX,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_minutes": train_minutes,
        "em": em,
        "f1": f1,
        "rougeL": rL,
    }


# ============================
# 12) CSV + summary
# ============================
def write_csv(rows: List[Dict[str, float]], path: str):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: List[Dict[str, float]]):
    def stats(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    ems = [r["em"] for r in rows]
    f1s = [r["f1"] for r in rows]
    rls = [r["rougeL"] for r in rows]
    mins = [r["train_minutes"] for r in rows]
    be = [r["best_epoch"] for r in rows]

    s_em = stats(ems)
    s_f1 = stats(f1s)
    s_rl = stats(rls)
    s_min = stats(mins)

    print("\nSummary across seeds")
    print(f"EM       mean={s_em['mean']:.4f} std={s_em['std']:.4f} min={s_em['min']:.4f} max={s_em['max']:.4f}")
    print(f"F1       mean={s_f1['mean']:.4f} std={s_f1['std']:.4f} min={s_f1['min']:.4f} max={s_f1['max']:.4f}")
    print(f"ROUGE-L  mean={s_rl['mean']:.4f} std={s_rl['std']:.4f} min={s_rl['min']:.4f} max={s_rl['max']:.4f}")
    print(f"Minutes  mean={s_min['mean']:.2f} std={s_min['std']:.2f} min={s_min['min']:.2f} max={s_min['max']:.2f}")
    print(f"Best epoch per seed: {be}")


# ============================
# 13) MAIN
# ============================
def main():
    if not Path(REQ_TEST_JSON).exists():
        raise FileNotFoundError(f"Missing {REQ_TEST_JSON}. Put it in the same folder as this script.")

    ensure_embedding_cache()

    rows = []
    for seed in SEEDS:
        row = train_one_seed(seed)
        rows.append(row)
        print(
            f"[seed={seed}] TEST RE-120: EM={row['em']:.4f} F1={row['f1']:.4f} ROUGE-L={row['rougeL']:.4f} "
            f"| best_epoch={row['best_epoch']} best_val_loss={row['best_val_loss']:.4f} "
            f"| train_min={row['train_minutes']:.2f}"
        )

    write_csv(rows, RESULTS_CSV)
    print(f"\nSaved per-seed results to: {RESULTS_CSV}")
    summarize(rows)
    print("\nDone.")


if __name__ == "__main__":
    print("SCRIPT START:", __file__)
    main()
