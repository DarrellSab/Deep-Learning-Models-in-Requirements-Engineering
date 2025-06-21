from transformers import BertTokenizerFast, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from torchinfo import summary

import psutil
from rouge_score import rouge_scorer
import time

# -----------------------------
# 1. Configuring the device for computation (GPU if available, otherwise CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# -----------------------------
# 2. Tokenizer and BERT model
# -----------------------------
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
max_len = 384


def prepare_features(example):
    encoding = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True
    )

    # Check if there is at least one answer
    if len(example["answers"]["answer_start"]) == 0:
        # Jei nėra atsakymo, grąžiname start/end = 0 (arba pagal poreikį)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "start": 0,
            "end": 0
        }

    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])
    offsets = encoding["offset_mapping"]

    start_token = end_token = 0
    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = idx
        if start < end_char <= end:
            end_token = idx
            break

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "start": start_token,
        "end": end_token
    }


# -----------------------------
# 3. Data loading and filtering
# -----------------------------
dataset = load_dataset("squad_v2")
train_data = dataset["train"].select(range(70000))
features = [prepare_features(ex) for ex in train_data]
train_split, val_split = train_test_split(features, test_size=0.2, random_state=42)


# -----------------------------
# 4. Bert model class
# ----------------------------- 
class BertQAModel(nn.Module):
    def __init__(self, model_name='bert-base-cased'):
        super(BertQAModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.start_logits = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_logits = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        start_logits = self.start_logits(sequence_output).squeeze(-1)  # (batch_size, seq_len)
        end_logits = self.end_logits(sequence_output).squeeze(-1)  # (batch_size, seq_len)

        return start_logits, end_logits


# -----------------------------
# 5. Dataset class
# ----------------------------- 
class QADataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        return {
            'input_ids': torch.tensor(item["input_ids"]),
            'attention_mask': torch.tensor(item["attention_mask"]),
            'start_positions': torch.tensor(item["start"]),
            'end_positions': torch.tensor(item["end"])
        }


train_dataset = QADataset(train_split)
val_dataset = QADataset(val_split)
print(f"CPU and RAM usage before train_loader: {psutil.virtual_memory().used / 1e9:.2f} GB")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# 6. Model training
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertQAModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()


def train_model(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        loss_start = loss_fn(start_logits, start_positions)
        loss_end = loss_fn(end_logits, end_positions)
        loss = (loss_start + loss_end) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        pred_start = torch.argmax(start_logits, dim=1)
        pred_end = torch.argmax(end_logits, dim=1)

        correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
        total_correct += correct
        total_examples += input_ids.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy

# -----------------------------
# 7. Outputs metrics and evaluates model performance during training without modifying weights
# -----------------------------

def evaluate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            loss_start = loss_fn(start_logits, start_positions)
            loss_end = loss_fn(end_logits, end_positions)
            loss = (loss_start + loss_end) / 2

            total_loss += loss.item()

            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)

            correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
            total_correct += correct
            total_examples += input_ids.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


print(f"Monitor CPU and RAM usage before train_model: {psutil.virtual_memory().used / 1e9:.2f} GB")
loss_fn = nn.CrossEntropyLoss()

total_training_time = 0.0
EPOCHS = 20
for epoch in range(EPOCHS):
    start_time = time.time()  # Pradžioje EPOCH ciklo
    torch.cuda.reset_peak_memory_stats()  # Resetinam GPU statistiką

    train_loss, train_acc = train_model(model, train_loader, optimizer, loss_fn)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn)

    epoch_time = time.time() - start_time  # Čia nebus klaidos!
    total_training_time += epoch_time

    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    print(f"Epoch {epoch + 1}/{EPOCHS}: "
          f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
    print(f"Epoch {epoch + 1} trukmė: {epoch_time / 60:.2f} min")
    print(f"Used VRAM memory (maks): {max_memory:.2f} GB")

print(f"\nTotal training time: {total_training_time / 60:.2f} min ({total_training_time / 3600:.2f} val)")

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertQAModel('bert-base-cased')
model.load_state_dict(torch.load("bert-qa-pytorch.pt", map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained("bert-qa-tokenizer")
print(f"CPU and RAM usage before testing: {psutil.virtual_memory().used / 1e9:.2f} GB")

# -----------------------------
# 8. Generate an answer from the question and context
# -----------------------------

# Compute ROUGE-L F1 score between the prediction and ground truth
def compute_rougeL(prediction, ground_truth):
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure


def get_answer(question, context, max_len=384):
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"])

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    if start_idx > end_idx:
        return ""

    tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()

# -----------------------------
# 9. Function to compute EM and F1 scores
# -----------------------------

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
# -----------------------------
# 10. Function to test the model using JSON format
# -----------------------------

def evaluate_bert_model(path, label=None, output_file="results.txt"):
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


print(f"CPU and RAM usage before evaluate_lstm_model: {psutil.virtual_memory().used / 1e9:.2f} GB")


evaluate_bert_model("requirements-test-30.json", label="Requirements test 30",
                    output_file="requirement_results-30-bert.txt")

print("BERT processing complete")