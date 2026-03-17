import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from tqdm import tqdm


# ============================
# 0) CONFIG
# ============================
SEEDS = [13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024]

SQUAD_TRAIN_LIMIT = 60000
MAX_LEN = 384
BATCH_SIZE = 4
LR = 3e-5
NUM_WORKERS = 0


MODEL_ID = "chandar-lab/NeoBERT"
MAX_LEN = 512

REQ_TEST_JSON = "requirements-test-120.json"
RESULTS_CSV = "results_10runs_neobert.csv"

USE_STRICT_DETERMINISM = True

# ---- Early stopping settings ----
EPOCHS_MAX = 10      # upper bound; will stop earlier
PATIENCE = 2         # stop after N consecutive non-improvements
MIN_DELTA = 0.0      # required improvement in val_loss to reset patience


# ============================
# 1) Device and AMP context
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

amp_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
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


def seed_worker_init(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================
# 3) Tokenizer
# ============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,local_files_only=True) # first running local should be False


# ============================
# 4) Prepare features (context-only mapping via sequence_ids)
# ============================
def prepare_features(example) -> Dict:
    enc = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True
    )

    seq_ids = enc.sequence_ids()

    if len(example["answers"]["answer_start"]) == 0:
        enc.pop("offset_mapping")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "start": 0,
            "end": 0
        }

    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])
    offsets = enc["offset_mapping"]

    start_token = end_token = 0

    for idx, ((s, e), sid) in enumerate(zip(offsets, seq_ids)):
        if sid != 1:
            continue
        if s <= start_char < e:
            start_token = idx
        if s < end_char <= e:
            end_token = idx
            break

    enc.pop("offset_mapping")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "start": start_token,
        "end": end_token
    }


# ============================
# 5) Dataset class
# ============================
class QADataset(Dataset):
    def __init__(self, features: List[Dict]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(item["start"], dtype=torch.long),
            "end_positions": torch.tensor(item["end"], dtype=torch.long),
        }


# ============================
# 6) NeoBERT QA model
# ============================
class NeoBERTQAModel(nn.Module):
    def __init__(self, model_name=MODEL_ID):
        super().__init__()

        self.neo_bert = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True) # first running local should be False

        self.start_head = nn.Linear(self.neo_bert.config.hidden_size, 1)
        self.end_head = nn.Linear(self.neo_bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.neo_bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state

        s = self.start_head(h).squeeze(-1)
        e = self.end_head(h).squeeze(-1)
        return s, e


# ============================
# 7) Training + validation loops
# ============================
loss_fn = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            s_log, e_log = model(input_ids, attention_mask)
            loss = (loss_fn(s_log, start_positions) + loss_fn(e_log, end_positions)) / 2.0

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        pred_s = torch.argmax(s_log, dim=1)
        pred_e = torch.argmax(e_log, dim=1)
        correct += ((pred_s == start_positions) & (pred_e == end_positions)).sum().item()
        total += input_ids.size(0)

    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def validate(model, loader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        with amp_ctx:
            s_log, e_log = model(input_ids, attention_mask)
            loss = (loss_fn(s_log, start_positions) + loss_fn(e_log, end_positions)) / 2.0

        total_loss += float(loss.item())
        pred_s = torch.argmax(s_log, dim=1)
        pred_e = torch.argmax(e_log, dim=1)
        correct += ((pred_s == start_positions) & (pred_e == end_positions)).sum().item()
        total += input_ids.size(0)

    return total_loss / max(1, len(loader)), correct / max(1, total)


# ============================
# 8) Metrics (EM/F1/ROUGE-L)
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
# 9) Span decoding for RE evaluation (context-only)
# ============================
@torch.no_grad()
def get_answer(model: NeoBERTQAModel, question: str, context: str) -> str:
    enc = tokenizer(
        question, context,
        truncation="only_second",
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = enc.pop("offset_mapping")[0]
    seq_ids = enc.sequence_ids(0)

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with amp_ctx:
        s_log, e_log = model(input_ids, attention_mask)

    s = int(torch.argmax(s_log, dim=1).item())
    e = int(torch.argmax(e_log, dim=1).item())
    if s > e:
        return ""

    if seq_ids[s] != 1 or seq_ids[e] != 1:
        return ""

    s0 = int(offsets[s][0])
    e1 = int(offsets[e][1])
    if e1 <= s0 or s0 < 0 or e1 < 0:
        return ""

    return context[s0:e1].strip()


def evaluate_re_json(model: NeoBERTQAModel, path: str) -> Tuple[float, float, float]:
    data = json.load(open(path, "r", encoding="utf-8"))
    em_sum = f1_sum = r_sum = 0.0

    for ex in data:
        pred = get_answer(model, ex["question"], ex["context"])
        gold = ex["expected_answer"]

        em_sum += float(normalize_text(pred) == normalize_text(gold))
        f1_sum += token_f1(pred, gold)
        r_sum += rouge_l(pred, gold)

    n = len(data)
    return em_sum / n, f1_sum / n, r_sum / n


# ============================
# 10) Build SQuAD features once
# ============================
def build_squad_features(limit: int) -> List[Dict]:
    ds = load_dataset("squad_v2")
    train = ds["train"].select(range(limit))

    filtered = [ex for ex in train if len(ex["answers"]["answer_start"]) > 0]
    print(f"SQuAD train records used (answerable): {len(filtered)} (from {limit})")

    feats = [prepare_features(ex) for ex in tqdm(filtered, desc="Preparing features")]
    return feats


# ============================
# 11) CSV output
# ============================
def write_csv(rows: List[Dict], path: str):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================
# 12) Run one seed with Early Stopping
# ============================
def run_one_seed(features: List[Dict], seed: int) -> Dict:
    set_seed(seed, strict=USE_STRICT_DETERMINISM)

    train_split, val_split = train_test_split(features, test_size=0.2, random_state=seed)

    train_ds = QADataset(train_split)
    val_ds = QADataset(val_split)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker_init,
        generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker_init,
        generator=g
    )


    model = NeoBERTQAModel(MODEL_ID).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = f"best_bert_seed_{seed}.pt"

    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        tr_loss, tr_span_acc = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_span_acc = validate(model, val_loader)

        print(f"[seed={seed}] epoch {epoch+1}/{EPOCHS_MAX} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"train_span_acc={tr_span_acc:.4f} val_span_acc={va_span_acc:.4f}")

        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f"[seed={seed}] New best checkpoint saved at epoch {best_epoch} (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[seed={seed}] Early stopping at epoch {epoch+1} "
                      f"(best epoch={best_epoch}, best val_loss={best_val:.4f})")
                break

    train_minutes = (time.time() - t0) / 60.0

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    em, f1, rL = evaluate_re_json(model, REQ_TEST_JSON)

    return {
        "seed": seed,
        "epochs_max": EPOCHS_MAX,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_minutes": train_minutes,
        "em": em,
        "f1": f1,
        "rougeL": rL
    }


# ============================
# 13) MAIN
# ============================
def main():
    if not Path(REQ_TEST_JSON).exists():
        raise FileNotFoundError(f"Missing {REQ_TEST_JSON} in the working directory.")

    print("Building SQuAD features once (reused across seeds)...")
    features = build_squad_features(SQUAD_TRAIN_LIMIT)

    rows = []
    for seed in SEEDS:
        row = run_one_seed(features, seed)
        rows.append(row)
        print(f"[seed={seed}] TEST RE-120: EM={row['em']:.4f} F1={row['f1']:.4f} ROUGE-L={row['rougeL']:.4f} "
              f"| best_epoch={row['best_epoch']} best_val_loss={row['best_val_loss']:.4f} "
              f"| train_min={row['train_minutes']:.2f}")

    write_csv(rows, RESULTS_CSV)
    print(f"\nSaved: {RESULTS_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()

