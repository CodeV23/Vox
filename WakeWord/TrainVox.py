import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ==========================================================
#                    DATASET
# ==========================================================
class WakewordDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]
        self.y = data["y"]

        # ⚙️ Normalize per-sample instead of globally
        # (prevents a few large values from dominating)
        for i in range(len(self.X)):
            self.X[i] = (self.X[i] - np.mean(self.X[i])) / (np.std(self.X[i]) + 1e-6)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # [1, 40, T]
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ==========================================================
#                    MODEL (CRNN)
# ==========================================================
class VoxCRNN(nn.Module):
    def __init__(self, n_mels=40, rnn_units=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),                # higher dropout for regularization
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),
        )

        self.gru = nn.GRU(
            input_size=32 * (n_mels // 4),
            hidden_size=rnn_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(rnn_units * 2, 1)

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last timestep
        return torch.sigmoid(self.fc(out))  # [B, 1]


# ==========================================================
#                    TRAINING LOOP
# ==========================================================
def train_and_freeze(train_loader, val_loader, device):
    model = VoxCRNN().to(device)
    crit = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularization

    best_val = 0.0
    patience = 3
    wait = 0
    best_path = "voxcrnn_best.pth"

    print("🚂 Starting training...")

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(1)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).squeeze().cpu().numpy()
                preds.extend((out > 0.5).astype(int).tolist())
                labels.extend(yb.int().tolist())

        f1 = f1_score(labels, preds)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_f1={f1:.4f}")

        # ---------------- Early stopping ----------------
        if f1 > best_val:
            best_val = f1
            wait = 0
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved new best model → {best_path}")
        else:
            wait += 1
            if wait >= patience:
                print("🛑 Early stopping.")
                break

    print(f"\n✅ Training complete. Best F1={best_val:.4f}")

    # ==========================================================
    # AUTO-FREEZE / EXPORT TO TORCHSCRIPT
    # ==========================================================
    print("\n🧊 Freezing model for fast inference...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dummy = torch.randn(1, 1, 40, 98).to(device)
    traced = torch.jit.trace(model, dummy)
    frozen_path = "voxcrnn_frozen.pt"
    traced.save(frozen_path)
    print(f"✅ Frozen TorchScript model saved → {frozen_path}\n")

    return model


# ==========================================================
#                    MAIN
# ==========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected device: {device}")

    train_ds = WakewordDataset("features_train.npz")
    val_ds = WakewordDataset("features_val.npz")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2, pin_memory=True)

    torch.backends.cudnn.benchmark = True
    train_and_freeze(train_loader, val_loader, device)


if __name__ == "__main__":
    main()
