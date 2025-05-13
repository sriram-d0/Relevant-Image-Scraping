import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from tqdm import tqdm
import re

# ----- Dataset Prep -----
class ImgDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.X = torch.tensor(vectorizer.transform(texts).toarray(), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def clean_text(entry):
    src = entry.get("src", "").lower()
    text = re.sub(r"<[^>]+>", " ", entry.get("text", "").lower())
    return f"{text} {src}"

# Load JSON
with open("annotation.json") as f:
    data = json.load(f)

texts, labels = [], []
for e in data:
    if "label" in e:
        texts.append(clean_text(e))
        labels.append(e["label"])

# Vectorize
vectorizer = TfidfVectorizer(max_features=2000)
vectorizer.fit(texts)

# Split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

train_set = ImgDataset(X_train, y_train, vectorizer)
test_set = ImgDataset(X_test, y_test, vectorizer)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16)

# ----- TextCNN Model -----
class TextCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(input_dim=2000).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ----- Training -----
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# ----- Eval -----
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_true.extend(y.tolist())

print("\n📊 Classification Report:")
print(classification_report(all_true, all_preds))
