import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time
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

from transformers import BertTokenizerFast, BertModel, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer

# ============================
# 0) CONFIG
# ============================
SEEDS = [13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024]

SQUAD_TRAIN_LIMIT = 60000

MAX_LEN = 384
BATCH_SIZE_TRAIN = 16

NUM_WORKERS = 4
PIN_MEMORY = True


LR_BERT = 3e-5
LR_LSTM = 1e-3


UNFREEZE_LAST_N_LAYERS = 2

REQ_TEST_JSON = "requirements-test-120.json"
RESULTS_CSV = "results_partial_finetune_lstm.csv"

USE_STRICT_DETERMINISM = True

# ---- Early stopping ----
EPOCHS_MAX = 10
PATIENCE = 2
MIN_DELTA = 0.0

# ============================
# 1) Device + AMP context
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


amp_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
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
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================
# 3) Tokenizer
# ============================
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


# ============================
# 4) Data Loading & Processing
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


class SQuADTokenizedDataset(Dataset):
    """
    Ši klasė pakeičia seną 'DiskDataset'.
    Ji laiko duomenis RAM ir grąžina (input_ids, attention_mask, start, end).
    """

    def __init__(self, data: List[dict], tokenizer, max_len=384):
        self.samples = []

        # Pre-process all data to speed up training loop
        print("Tokenizing data...")

        questions = [ex["question"] for ex in data]
        contexts = [ex["context"] for ex in data]
        answers = [ex["answers"]["text"][0] for ex in data]
        starts = [ex["answers"]["answer_start"][0] for ex in data]


        encodings = tokenizer(
            questions,
            contexts,
            max_length=max_len,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offset_mapping = encodings.pop("offset_mapping")

        for i in range(len(data)):
            # Label alignment logic (from your original code)
            start_char = starts[i]
            end_char = start_char + len(answers[i])
            sequence_ids = encodings.sequence_ids(i)
            offsets = offset_mapping[i]

            # Find token start/end
            token_start_index = 0
            token_end_index = 0

            # Start loop
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
                if idx >= len(sequence_ids): break
            context_start = idx

            if idx < len(sequence_ids):
                # End loop
                while sequence_ids[idx] == 1:
                    idx += 1
                    if idx >= len(sequence_ids): break
                context_end = idx - 1

                # If answer is not fully inside context, label is (0,0) - CLS token
                # (SQuAD v2 answerable should be inside, but truncation might cut it)
                if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                    token_start_index = 0
                    token_end_index = 0
                else:
                    # Find start
                    idx = context_start
                    while idx <= context_end and offsets[idx][0] <= start_char:
                        idx += 1
                    token_start_index = idx - 1

                    # Find end
                    idx = context_end
                    while idx >= context_start and offsets[idx][1] >= end_char:
                        idx -= 1
                    token_end_index = idx + 1

            self.samples.append({
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
                "start_positions": torch.tensor(token_start_index, dtype=torch.long),
                "end_positions": torch.tensor(token_end_index, dtype=torch.long)
            })

        print(f"Tokenization complete. Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================
# 5) NEW MODEL: BERT + LSTM (Integrated) 3 LAYERS
# ============================
class QALSTM_PartialFineTune(nn.Module):
    def __init__(self, model_name='bert-base-cased',
                 lstm1_size=1024, lstm2_size=512, lstm3_size=512, dense_size=512,
                 unfreeze_last_n=2):
        super(QALSTM_PartialFineTune, self).__init__()


        self.bert = BertModel.from_pretrained(model_name)


        for param in self.bert.parameters():
            param.requires_grad = False


        if unfreeze_last_n > 0:
            print(f"--> Unfreezing last {unfreeze_last_n} BERT encoder layers...")
            layers_to_unfreeze = self.bert.encoder.layer[-unfreeze_last_n:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("--> BERT is completely frozen.")


        self.lstm1 = nn.LSTM(768, lstm1_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(lstm1_size, lstm2_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(lstm2_size, lstm3_size, batch_first=True)
        self.dropout3 = nn.Dropout(0.1)

        #  Heads
        self.projection = nn.Linear(lstm3_size, dense_size)
        self.start_out = nn.Linear(dense_size, 1)
        self.end_out = nn.Linear(dense_size, 1)

    def forward(self, input_ids, attention_mask):
        # BERT Pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq, 768]

        # LSTM Pass
        x, _ = self.lstm1(sequence_output)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)

        # Projection
        x = F.relu(self.projection(x))

        # Outputs
        start_logits = self.start_out(x).squeeze(-1)
        end_logits = self.end_out(x).squeeze(-1)

        return start_logits, end_logits


# ============================
# 6) Metrics
# ============================
def normalize_text(t: str) -> str:
    return " ".join(t.lower().split())


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_text(pred).split()
    gt = normalize_text(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0

    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0: return 0.0

    p = num_same / len(pt)
    r = num_same / len(gt)
    return 2 * p * r / (p + r)


def rouge_l(pred: str, gold: str) -> float:
    return float(scorer.score(gold, pred)["rougeL"].fmeasure)


# ============================
# 7) Evaluation Logic (Updated for new model)
# ============================
@torch.no_grad()
def get_model_answer(q: str, ctx: str, model: nn.Module) -> str:

    enc = tokenizer(
        q, ctx,
        max_length=MAX_LEN,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    ).to(device)

    offs = enc.pop("offset_mapping")[0]
    seq_ids = enc.sequence_ids(0)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with amp_ctx:

        s_log, e_log = model(input_ids, attention_mask)

    s = int(torch.argmax(s_log, dim=-1).item())
    e = int(torch.argmax(e_log, dim=-1).item())

    if s > e: return ""

    if seq_ids[s] != 1 or seq_ids[e] != 1: return ""

    s0 = int(offs[s][0])
    e1 = int(offs[e][1])
    if s0 < 0 or e1 < 0 or e1 <= s0: return ""

    return ctx[s0:e1].strip()


def evaluate_requirements_json(path: str, model: nn.Module) -> Tuple[float, float, float]:
    data = json.load(open(path, "r", encoding="utf-8"))
    em_sum, f1_sum, r_sum = 0.0, 0.0, 0.0

    model.eval()
    for ex in data:
        pred = get_model_answer(ex["question"], ex["context"], model)
        gold = ex["expected_answer"]

        em_sum += float(normalize_text(pred) == normalize_text(gold))
        f1_sum += token_f1(pred, gold)
        r_sum += rouge_l(pred, gold)

    n = len(data)
    return em_sum / n, f1_sum / n, r_sum / n


# ============================
# 8) Training Loop
# ============================
loss_fn = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        start_pos = batch["start_positions"].to(device, non_blocking=True)
        end_pos = batch["end_positions"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            s_log, e_log = model(input_ids, attention_mask)
            loss = loss_fn(s_log, start_pos) + loss_fn(e_log, end_pos)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Accuracy tracking
        pred_s = torch.argmax(s_log, dim=1)
        pred_e = torch.argmax(e_log, dim=1)
        correct += ((pred_s == start_pos) & (pred_e == end_pos)).sum().item()
        total += input_ids.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        start_pos = batch["start_positions"].to(device, non_blocking=True)
        end_pos = batch["end_positions"].to(device, non_blocking=True)

        with amp_ctx:
            s_log, e_log = model(input_ids, attention_mask)
            loss = loss_fn(s_log, start_pos) + loss_fn(e_log, end_pos)

        total_loss += loss.item()
        pred_s = torch.argmax(s_log, dim=1)
        pred_e = torch.argmax(e_log, dim=1)
        correct += ((pred_s == start_pos) & (pred_e == end_pos)).sum().item()
        total += input_ids.size(0)

    return total_loss / len(loader), correct / total


# ============================
# 9) Main Seed Loop
# ============================
def train_one_seed(seed: int, full_data_list: List[dict]) -> Dict[str, float]:
    set_seed(seed, strict=USE_STRICT_DETERMINISM)
    g = make_torch_generator(seed)



    random.shuffle(full_data_list)
    n_train = int(0.8 * len(full_data_list))
    train_raw = full_data_list[:n_train]
    val_raw = full_data_list[n_train:]

    train_ds = SQuADTokenizedDataset(train_raw, tokenizer, max_len=MAX_LEN)
    val_ds = SQuADTokenizedDataset(val_raw, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker_init, generator=g, persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )


    model = QALSTM_PartialFineTune(unfreeze_last_n=UNFREEZE_LAST_N_LAYERS).to(device)


    optimizer = torch.optim.AdamW([

        {'params': [p for p in model.bert.parameters() if p.requires_grad], 'lr': LR_BERT},


        {'params': model.lstm1.parameters(), 'lr': LR_LSTM},
        {'params': model.lstm2.parameters(), 'lr': LR_LSTM},
        {'params': model.lstm3.parameters(), 'lr': LR_LSTM},
        {'params': model.projection.parameters(), 'lr': LR_LSTM},
        {'params': model.start_out.parameters(), 'lr': LR_LSTM},
        {'params': model.end_out.parameters(), 'lr': LR_LSTM}
    ])

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = f"best_model_seed_{seed}.pt"

    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_acc = validate(model, val_loader)

        print(f"[seed={seed}] ep {epoch + 1} | T_loss={tr_loss:.4f} V_loss={va_loss:.4f} | V_acc={va_acc:.4f}")

        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping @ epoch {epoch + 1}")
                break

    train_minutes = (time.time() - t0) / 60.0

    # Load best & Evaluate
    model.load_state_dict(torch.load(best_path, map_location=device))
    em, f1, rL = evaluate_requirements_json(REQ_TEST_JSON, model)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_minutes": train_minutes,
        "em": em, "f1": f1, "rougeL": rL
    }


def main():
    if not Path(REQ_TEST_JSON).exists():
        raise FileNotFoundError(f"Missing {REQ_TEST_JSON}")

    # 1. Load Data once
    raw_data = load_filtered_squad(SQUAD_TRAIN_LIMIT)

    rows = []
    for seed in SEEDS:
        print(f"\n--- START SEED {seed} ---")
        
        data_copy = list(raw_data)

        row = train_one_seed(seed, data_copy)
        rows.append(row)
        print(f"RESULT: F1={row['f1']:.4f} | Time={row['train_minutes']:.1f}m")

    # Save Results
    import csv
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("\nDONE.")


if __name__ == "__main__":
    main()