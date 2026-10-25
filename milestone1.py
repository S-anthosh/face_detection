import random
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import matplotlib.pyplot as plt

ROOT = Path(r"D:\VSB-TUO\machine learning\kaggle_cache\datasets\hereisburak\pins-face-recognition\versions\1\105_classes_pins_dataset")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PinsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        dirs = sorted(d for d in Path(root).iterdir() if d.is_dir())
        self.classes = [d.name for d in dirs]
        self.samples = [(p, i) for i, d in enumerate(dirs) for p in d.glob("*.jpg")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 12 * 12, 256), nn.ReLU(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, dl, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    print("Device:", device)
    tf = T.Compose([T.Resize((96, 96)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    full = PinsDataset(ROOT, tf)
    num_classes = len(full.classes)
    print("Images:", len(full), "Classes:", num_classes)

    # stratified split
    by_class = {}
    for i, (_, y) in enumerate(full.samples):
        by_class.setdefault(y, []).append(i)
    random.seed(42)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        random.shuffle(idxs)
        k = int(0.8 * len(idxs))
        train_idx += idxs[:k]
        val_idx += idxs[k:]

    train_dl = DataLoader(Subset(full, train_idx), batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(Subset(full, val_idx), batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

    # visualize one batch
    x, y = next(iter(train_dl))
    print("Batch:", x.shape, y.shape)
    grid = (x[:8] * 0.5 + 0.5).permute(0, 2, 3, 1)
    fig, axs = plt.subplots(1, 8, figsize=(20, 3))
    for a, im, lb in zip(axs, grid, y[:8]):
        a.imshow(im)
        a.set_title(full.classes[lb][5:17])
        a.axis("off")
    plt.savefig("batch.png")
    plt.show()

    criterion = nn.CrossEntropyLoss()

    # overfit sanity check (16 samples)
    model = BaselineCNN(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tiny_dl = DataLoader(Subset(full, train_idx[:16]), batch_size=16, shuffle=True)
    for ep in range(100):
        loss, acc = run_epoch(model, tiny_dl, criterion, opt)
        if ep % 10 == 0:
            print(f"[overfit] ep {ep}: loss {loss:.4f} acc {acc:.2f}")
        if loss < 1e-3 and acc == 1.0:
            print(f"[overfit] done at ep {ep}: loss {loss:.5f} acc {acc:.2f}")
            break

    # full baseline training
    model = BaselineCNN(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}
    for ep in range(15):
        tr = run_epoch(model, train_dl, criterion, opt)
        va = run_epoch(model, val_dl, criterion)
        history["tr_loss"].append(tr[0]); history["tr_acc"].append(tr[1])
        history["va_loss"].append(va[0]); history["va_acc"].append(va[1])
        print(f"ep {ep}: train loss {tr[0]:.3f} acc {tr[1]:.3f} | val loss {va[0]:.3f} acc {va[1]:.3f}")

    # curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["tr_loss"], label="train"); a1.plot(history["va_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(history["tr_acc"], label="train"); a2.plot(history["va_acc"], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    plt.tight_layout()
    plt.savefig("curves.png")
    plt.show()
    
if __name__ == "__main__":
    main()