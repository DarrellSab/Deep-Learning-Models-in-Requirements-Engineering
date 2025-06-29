import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import psutil
import shutil
import json
from pathlib import Path
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
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
bert_model = BertModel.from_pretrained("bert-base-cased").to(device)
bert_model.eval()

# -----------------------------
# 3. Data loading and filtering
# -----------------------------
dataset = load_dataset("squad_v2")
train_data = dataset["train"].select(range(70000))  # Galima keisti į daugiau

filtered_data = [
    ex for ex in train_data
    if isinstance(ex["answers"]["text"], list) and len(ex["answers"]["text"]) > 0
       and isinstance(ex["answers"]["answer_start"], list) and len(ex["answers"]["answer_start"]) > 0
]

print(f"Records used: {len(filtered_data)}")


# -----------------------------
# 4. Generating and streaming BERT embeddings to disk
# -----------------------------
def generate_embeddings_stream(data, tokenizer, model, max_len=256, batch_size=16, out_dir="embeddings"):
    device = next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)
    embedding_idx = 0

    for batch_start in tqdm(range(0, len(data), batch_size), desc="Generating embeddings", disable=True):
        batch = data[batch_start:batch_start + batch_size]
        questions = [ex["question"] for ex in batch]
        contexts = [ex["context"] for ex in batch]
        answers = [ex["answers"]["text"][0] for ex in batch]
        starts = [ex["answers"]["answer_start"][0] for ex in batch]

        encodings = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offset_mapping = encodings.pop("offset_mapping").tolist()
        sequence_ids = [encodings.sequence_ids(i) for i in range(len(batch))]
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            last_hidden_states = outputs.last_hidden_state.cpu().numpy()

        for i in range(len(batch)):
            seq_ids = sequence_ids[i]
            offsets = offset_mapping[i]
            start_char = starts[i]
            end_char = start_char + len(answers[i])
            start_token = end_token = 0

            for idx, (offset, seq_id) in enumerate(zip(offsets, seq_ids)):
                if seq_id != 1:
                    continue
                if offset[0] <= start_char < offset[1]:
                    start_token = idx
                if offset[0] < end_char <= offset[1]:
                    end_token = idx

            np.save(os.path.join(out_dir, f"X_{embedding_idx}.npy"), last_hidden_states[i])
            np.save(os.path.join(out_dir, f"y_start_{embedding_idx}.npy"), np.array([start_token]))
            np.save(os.path.join(out_dir, f"y_end_{embedding_idx}.npy"), np.array([end_token]))
            embedding_idx += 1

    print(f"Embeddings have been generated and saved {embedding_idx} records saved to folder '{out_dir}'.")


# Execution
generate_embeddings_stream(
    data=filtered_data,
    tokenizer=tokenizer,
    model=bert_model,
    max_len=128,
    batch_size=32,
    out_dir="embeddings"
)

print(f"CPU and RAM usage after generate_embeddings: {psutil.virtual_memory().used / 1e9:.2f} GB")


# -----------------------------
# 5. Dataset class
# -----------------------------
class QADiskDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.indices = [int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(data_dir) if
                        fname.startswith("X_")]
        self.indices.sort()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        x = np.load(os.path.join(self.data_dir, f"X_{index}.npy"))
        y_start = np.load(os.path.join(self.data_dir, f"y_start_{index}.npy"))[0]
        y_end = np.load(os.path.join(self.data_dir, f"y_end_{index}.npy"))[0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_start, dtype=torch.long), torch.tensor(y_end,
                                                                                                           dtype=torch.long)


# -----------------------------
# 6. LSTM model
# -----------------------------
class QALSTMModel(nn.Module):
    def __init__(self, input_size=768, lstm1_size=1024, lstm2_size=512, lstm3_size=512, dense_size=512):
        super(QALSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, lstm1_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(lstm1_size, lstm2_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(lstm2_size, lstm3_size, batch_first=True)
        self.dropout3 = nn.Dropout(0.1)
        self.projection = nn.Linear(lstm3_size, dense_size)
        self.start_out = nn.Linear(dense_size, 1)
        self.end_out = nn.Linear(dense_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x = F.relu(self.projection(x))
        start_logits = self.start_out(x).squeeze(-1)
        end_logits = self.end_out(x).squeeze(-1)
        return start_logits, end_logits


# -----------------------------
# 7. Model training
# -----------------------------
dataset = QADiskDataset(data_dir="embeddings")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = QALSTMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)  # prieš tai lr=1e-5
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 10
best_val_loss = float("inf")
total_training_time = 0.0

for epoch in range(EPOCHS):
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    model.train()
    total_loss = 0
    correct_start_train = 0
    correct_end_train = 0
    total_train = 0

    for X_batch, y_start_batch, y_end_batch in train_loader:
        X_batch = X_batch.to(device)
        y_start_batch = y_start_batch.to(device)
        y_end_batch = y_end_batch.to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(X_batch)
        loss_start = loss_fn(start_logits, y_start_batch)
        loss_end = loss_fn(end_logits, y_end_batch)
        loss = 1.5 * loss_start + 1.0 * loss_end
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Train Accuracy counting
        start_preds = torch.argmax(start_logits, dim=1)
        end_preds = torch.argmax(end_logits, dim=1)
        correct_start_train += (start_preds == y_start_batch).sum().item()
        correct_end_train += (end_preds == y_end_batch).sum().item()
        total_train += y_start_batch.size(0)

    train_acc_start = correct_start_train / total_train
    train_acc_end = correct_end_train / total_train
    train_acc = (train_acc_start + train_acc_end) / 2

    model.eval()
    val_loss = 0
    correct_start = 0
    correct_end = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_start_batch, y_end_batch in val_loader:
            X_batch = X_batch.to(device)
            y_start_batch = y_start_batch.to(device)
            y_end_batch = y_end_batch.to(device)
            start_logits, end_logits = model(X_batch)
            val_loss_start = loss_fn(start_logits, y_start_batch)
            val_loss_end = loss_fn(end_logits, y_end_batch)
            val_loss += (1.5 * val_loss_start + 1.0 * val_loss_end).item()
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            correct_start += (start_preds == y_start_batch).sum().item()
            correct_end += (end_preds == y_end_batch).sum().item()
            total += y_start_batch.size(0)

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    acc_start = correct_start / total
    acc_end = correct_end / total
    avg_val_acc = (acc_start + acc_end) / 2

    epoch_time = time.time() - start_time
    total_training_time += epoch_time
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
    print(f"Used VRAM memory (maks): {max_memory:.2f} GB")
    print(f"Epoch duration: {epoch_time / 60:.2f} min")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_qa_lstm_model.pt")
        print("Model saved.")
print(f"\nTotal training time: {total_training_time / 60:.2f} min ({total_training_time / 3600:.2f} val)")

shutil.rmtree("embeddings")
print("Embeddings' directory deleted.")
# -------------------------------------
# 8. Loading the trained model and BERT encode
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QALSTMModel().to(device)
model.load_state_dict(torch.load("best_qa_lstm_model.pt", map_location=device))
model.eval()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
bert_model = BertModel.from_pretrained("bert-base-cased").to(device)
bert_model.eval()


# -------------------------------------
# 9. Generate an answer from the question and context
# -------------------------------------

# Compute ROUGE-L F1 score between the prediction and ground truth
def compute_rougeL(prediction, ground_truth):
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure


def get_lstm_answer(question, context, max_len=384):
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_len,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.sequence_ids(0)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        bert_output = bert_model(**inputs)
        embeddings = bert_output.last_hidden_state  # [1, seq_len, hidden_size]

        start_logits, end_logits = model(embeddings)

        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()

    # Validate that the indices are valid
    if start_index > end_index or sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
        return ""

    start_char = offset_mapping[start_index][0]
    end_char = offset_mapping[end_index][1]
    return context[start_char:end_char].strip()


# -------------------------------------
# 10. Function to compute EM and F1 scores
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
# 11. Function to test the model using JSON format
# -------------------------------------
def evaluate_lstm_model(path, label=None, output_file="results.txt"):
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

        predicted = get_lstm_answer(question, context)
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

# -------------------------------------
# 12. Executing the test function using a JSON-formatted test
# -------------------------------------


evaluate_lstm_model("requirements-test-30.json", label="Requirements test 30",
                    output_file="requirement_results-30-LSTM.txt")

print("Bert+LSTM process finished")
