# train_1d_classifier_fixed.py
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ---------------------------
# Dataset and Model definitions (top-level, importable)
# ---------------------------
sample_size = int(1e6)
n_samples = int(500)
periods = [1000, 2000, 5000, 7000, 10000]

data_path = "training_set.dat"
labels_path = "labels.npy"

class LargeSignalDataset(Dataset):
    def __init__(self, data_file, labels_file):
        # open memmap in read-only mode to avoid copying into RAM
        self.data_file = data_file
        self.X = np.memmap(data_file, dtype='float32', mode='r', shape=(n_samples, sample_size))
        self.y = np.load(labels_file)
        self.periods = np.array(periods)
        self.period_to_idx = {p: idx for idx, p in enumerate(self.periods)}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        arr = self.X[idx].astype('float32')
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, L]
        label_value = int(self.y[idx])
        label_idx = self.period_to_idx[label_value]
        return tensor, int(label_idx)

class Downsample1DCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, stride=16, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=31, stride=16, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Helper to create memmap if missing
# ---------------------------
def create_memmap_if_missing():
    if os.path.exists(data_path) and os.path.exists(labels_path):
        print("Dataset memmap and labels already exist; skipping creation.")
        return

    print("Creating dataset memmap on disk (this may take a while)...")
    X_mm = np.memmap(data_path, dtype='float32', mode='w+', shape=(n_samples, sample_size))
    y = np.empty(n_samples, dtype=np.int32)

    x_event = np.arange(-100, 100)
    pos = 2000 * np.exp(-0.5 * ((x_event + 70) / 12) ** 2)
    neg = -2000 * np.exp(-0.5 * ((x_event - 70) / 12) ** 2)
    event = (pos + neg).astype('float32')

    # fill with noise
    for i in tqdm(range(n_samples), desc="Filling noise"):
        X_mm[i, :] = np.random.normal(0, 5, sample_size).astype('float32')

    per_class = n_samples // len(periods)
    idx_base = 0
    for j, set_period in enumerate(periods):
        for i_local in tqdm(range(per_class), desc=f"Adding signals for period {set_period}"):
            global_i = idx_base + i_local
            rng = np.random.RandomState(global_i + 12345)
            shift0 = np.random.randint(400, set_period)
            # add event periodically
            for k in range(sample_size // set_period):
                idx = k * set_period + shift0
                if idx - 100 >= 0 and idx + 99 < sample_size:
                    X_mm[global_i, idx - 100: idx + 100] += event
            y[global_i] = set_period
        idx_base += per_class

    remainder = n_samples - per_class * len(periods)
    if remainder > 0:
        j = len(periods) - 1
        set_period = periods[j]
        for r in tqdm(range(remainder), desc="Adding remainder signals"):
            global_i = idx_base + r
            rng = np.random.RandomState(global_i + 12345)
            shift0 = np.random.randint(400, set_period)
            for k in range(sample_size // set_period):
                idx = k * set_period + shift0
                if idx - 100 >= 0 and idx + 99 < sample_size:
                    X_mm[global_i, idx - 100: idx + 100] += event
            y[global_i] = set_period

    del X_mm
    np.save(labels_path, y)
    print("Dataset creation finished.")

# ---------------------------
# Main runtime: guarded so DataLoader workers can import module safely
# ---------------------------
if __name__ == "__main__":
    # Create dataset file if needed
    create_memmap_if_missing()

    # Device: use MPS/GPU/CPU auto detection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)

    # Instantiate dataset
    dataset = LargeSignalDataset(data_path, labels_path)

    # split train/val
    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [num_train, num_val])

    # DataLoader settings
    # NOTE: On macOS/MPS use num_workers=0 and pin_memory=False to avoid spawn/pinned memory issues.
    # If you run on Linux+CUDA, you can increase num_workers and set pin_memory=True.
    num_workers = 0      # safe default; increase on Linux/GPU
    pin_memory = False   # set True only for CUDA when beneficial

    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = Downsample1DCNN(n_classes=len(periods)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    idx_to_period = {i: p for i, p in enumerate(periods)}

    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{epochs} | Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "idx_to_period": idx_to_period
    }, "one_d_cnn_model.pth")
    print("Model saved to one_d_cnn_model.pth")
