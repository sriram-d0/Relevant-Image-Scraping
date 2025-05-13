from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import pandas as pd
import json
import re

# Load JSON
with open("annotation.json") as f:
    data = json.load(f)

def clean_text(entry):
    src = entry.get("src", "").lower()
    text = re.sub(r"<[^>]+>", " ", entry.get("text", "").lower())
    return f"{text} {src}"

texts, labels = [], []
for e in data:
    if "label" in e:
        texts.append(clean_text(e))
        labels.append(e["label"])

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_enc = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
test_enc = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt")

class HTMLDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_dataset = HTMLDataset(train_enc, y_train)
test_dataset = HTMLDataset(test_enc, y_test)

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./distilbert-img",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Eval
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(axis=1)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
