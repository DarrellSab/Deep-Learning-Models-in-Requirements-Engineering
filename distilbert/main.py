import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    get_scheduler
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import json
import numpy as np
from torchinfo import summary
import psutil
from rouge_score import rouge_scorer
from pathlib import Path
import time

# -----------------------------
# 1. Configuring the device for computation (GPU if available, otherwise CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# -----------------------------
# 2. Data loading and filtering
# -----------------------------
dataset = load_dataset("squad_v2")
train_data = dataset["train"].select(range(70000))
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
print(f"CPU and RAM usage after dataset: {psutil.virtual_memory().used / 1e9:.2f} GB")

# -----------------------------
# 3. Data preparing
# -----------------------------

def prepare_features(example):
    encoding = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = encoding["offset_mapping"][0].tolist()
    sequence_ids = encoding.sequence_ids(0)

    # Handle missing answer
    if len(example["answers"]["answer_start"]) == 0:
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "start_positions": 0,
            "end_positions": 0
        }

    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])

    start_token = end_token = 0
    found_start = found_end = False

    for idx, (offset, seq_id) in enumerate(zip(offset_mapping, sequence_ids)):
        if seq_id != 1:
            continue
        start, end = offset
        if start <= start_char < end and not found_start:
            start_token = idx
            found_start = True
        if start < end_char <= end and not found_end:
            end_token = idx
            found_end = True
        if found_start and found_end:
            break

    if not found_start:
        start_token = 0
    if not found_end:
        end_token = 0

    return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze(),
        "start_positions": start_token,
        "end_positions": end_token
    }
# -----------------------------
# 4. Model preparing and training
# -----------------------------

# Process all examples
encoded_data = [prepare_features(ex) for ex in train_data]
input_ids = torch.tensor([ex["input_ids"].tolist() for ex in encoded_data])
attention_mask = torch.tensor([ex["attention_mask"].tolist() for ex in encoded_data])
start_positions = torch.tensor([ex["start_positions"] for ex in encoded_data])
end_positions = torch.tensor([ex["end_positions"] for ex in encoded_data])

# Split into train and validation sets
dataset_torch = torch.utils.data.TensorDataset(input_ids, attention_mask, start_positions, end_positions)
train_size = int(0.8 * len(dataset_torch))
val_size = len(dataset_torch) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset_torch, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
print(f"CPU and RAM usage after train_loader: {psutil.virtual_memory().used / 1e9:.2f} GB")


model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased").to(device)

# Optimizer ir scheduler
total_training_time = 0.0
EPOCHS = 40
optimizer = AdamW(model.parameters(), lr=1e-5)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = nn.CrossEntropyLoss()


def train_model(model, data_loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        input_ids, attention_mask, start_positions, end_positions = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

        pred_start = torch.argmax(outputs.start_logits, dim=1)
        pred_end = torch.argmax(outputs.end_logits, dim=1)
        correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
        total_correct += correct
        total_examples += input_ids.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, start_positions, end_positions = [b.to(device) for b in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs.loss
            total_loss += loss.item()

            pred_start = torch.argmax(outputs.start_logits, dim=1)
            pred_end = torch.argmax(outputs.end_logits, dim=1)
            correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
            total_correct += correct
            total_examples += input_ids.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


# Train
model.train()
for epoch in range(EPOCHS):
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    train_loss, train_acc = train_model(model, train_loader, optimizer, lr_scheduler, loss_fn, device)
    val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)

    epoch_time = time.time() - start_time
    total_training_time += epoch_time

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"Epoch {epoch + 1}/{EPOCHS}: "
          f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
    print(f"Epoch {epoch + 1} duration: {epoch_time / 60:.2f} min")
    print(f"Used VRAM memory (maks): {max_memory:.2f} GB")

print(f"\nTotal training time: {total_training_time / 60:.2f} min ({total_training_time / 3600:.2f} val)")

# Save model
model.save_pretrained("distilbert_qa_model_v2_torch")
tokenizer.save_pretrained("distilbert_tokenizer_v2_torch")
# -------------------------------------
# 5. Load the model and tokenizer
# -------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert_qa_model_v2_torch").to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert_tokenizer_v2_torch")
print(f"CPU and RAM usage after tokenizer: {psutil.virtual_memory().used / 1e9:.2f} GB")

# -------------------------------------
# 6. Functions to generate an answer from the question and context
# -------------------------------------

# Compute ROUGE-L F1 score between the prediction and ground truth
def compute_rougeL(prediction, ground_truth):
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure


def get_answer(question, context):
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, dim=1).item()
    end_idx = torch.argmax(outputs.end_logits, dim=1).item()

    if start_idx > end_idx:
        return ""

    tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


# -------------------------------------
# 7. Function to compute EM and F1 scores
# -------------------------------------
def compute_em_and_f1(prediction, ground_truth):
    def normalize(text):
        return " ".join(text.lower().strip().split())

    pred = normalize(prediction)
    truth = normalize(ground_truth)
    em = int(pred == truth)

    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = set(pred_tokens) & set(truth_tokens)

    if not common:
        return em, 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1

# -------------------------------------
# 8. Function to test the model using JSON format
# -------------------------------------

def evaluate_distilbert_model(path, label=None, output_file="results.txt"):
    em_list = []
    f1_list = []
    rougeL_list = []
    output_lines = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error while reading {path}: {e}")
        return

    header = f"\nTesting: {label or path}\n{'-' * 40}"
    print(header)
    output_lines.append(header)

    for i, item in enumerate(test_data):
        question = item["question"]
        context = item["context"]
        expected = item["expected_answer"]

        predicted = get_answer(question, context)
        em, f1 = compute_em_and_f1(predicted, expected)
        rougeL = compute_rougeL(predicted, expected)

        em_list.append(em)
        f1_list.append(f1)
        rougeL_list.append(rougeL)

        result_str = (
            f"{i + 1}. Q: {question}\n"
            f"Model answer: {predicted}\n"
            f"Exact answer: {expected}\n"
            f"EM: {em}, F1: {round(f1, 3)}, ROUGE-L: {round(rougeL, 3)}\n"
        )
        print(result_str)
        output_lines.append(result_str)

    avg_em = sum(em_list) / len(em_list) if em_list else 0.0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    avg_rougeL = sum(rougeL_list) / len(rougeL_list) if rougeL_list else 0.0

    summary = (
        f"\nTotal results ({label or path}):\n"
        f" EM: {round(avg_em * 100, 2)}%\n"
        f"F1: {round(avg_f1 * 100, 2)}%\n"
        f"ROUGE-L: {round(avg_rougeL * 100, 2)}%\n"
    )
    print(summary)
    output_lines.append(summary)

    try:
        Path(output_file).write_text("\n".join(output_lines), encoding="utf-8")
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error while saving results: {e}")


print(f"CPU and RAM usage before: {psutil.virtual_memory().used / 1e9:.2f} GB")


evaluate_distilbert_model("requirements-test-30.json", label="Requirements test 30",
                          output_file="requirement_results_30-distilbert.txt")

print("Distilbert process finished")