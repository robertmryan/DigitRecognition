# mnist_cnn_mps.py
import os, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Device (MPS on Apple Silicon if available) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Reproducibility (best-effort on MPS) ---
# MPS is improving but not guaranteed bit-for-bit deterministic across runs.
torch.manual_seed(42)

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST normalization
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# I've already downloaded and want to make sure I'm using the exact same files I did before

# train_ds = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
# test_ds  = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

# --- Model ---
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)   # 28->26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  # 13->11 after pool
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (N,32,26,26)
        x = self.pool(x)               # (N,32,13,13)
        x = F.relu(self.conv2(x))      # (N,64,11,11)
        x = self.pool(x)               # (N,64,5,5)
        x = torch.flatten(x, 1)        # (N,1600)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SmallCNN().to(device)

# --- Optimizer & Scheduler ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# --- Train / Eval loops ---
def train_one_epoch(epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()

    acc = correct / total
    print(f"Epoch {epoch:02d} | train loss {loss_sum/total:.4f} | train acc {acc*100:.2f}%")

def evaluate():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
    return loss_sum/total, correct/total

# --- Run ---
start = time.time()
epochs = 12
for epoch in range(1, epochs+1):
    train_one_epoch(epoch)
    val_loss, val_acc = evaluate()
    print(f"           | val loss {val_loss:.4f} | val acc {val_acc*100:.2f}%")
    scheduler.step()

elapsed = time.time() - start
print(f"\nTraining finished in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes).")

# Save the trained weights
os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "artifacts/mnist_smallcnn_mps.pt")
print("Saved to artifacts/mnist_smallcnn_mps.pt")
