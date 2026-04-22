import torch
import torch.nn as nn

torch.manual_seed(42)

D = 8
N_TRAIN = 30
RELEVANT = [1, 8, 20]  # sparse positions

def generate_data(n_samples, N):
    X = torch.randn(n_samples, N, D)
    y = sum(X[:, p, 0] for p in RELEVANT if p < N)
    return X, (y > 0).float()

X_train, y_train = generate_data(5000, N_TRAIN)
X_test_id, y_test_id = generate_data(2000, N_TRAIN)
X_test_ood, y_test_ood = generate_data(2000, N=60)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, num_heads=1, batch_first=True)
        self.fc = nn.Linear(D, 1)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.fc(out.mean(1)).squeeze()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_TRAIN * D, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x.flatten(1)).squeeze()

def train(model, X, y, epochs=1000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
        if epoch % 250 == 0:
            with torch.no_grad():
                acc = ((model(X) > 0).float() == y).float().mean().item()
            print(f"  epoch {epoch}, train acc {acc:.3f}")

def acc(model, X, y):
    with torch.no_grad():
        return ((model(X) > 0).float() == y).float().mean().item()

print("Training transformer...")
transformer = Transformer()
train(transformer, X_train, y_train)

print("\nTraining MLP...")
mlp = MLP()
train(mlp, X_train, y_train)

print("\n=== Results ===")
print(f"Transformer - in-distribution:  {acc(transformer, X_test_id, y_test_id):.3f}")
print(f"MLP         - in-distribution:  {acc(mlp, X_test_id, y_test_id):.3f}")
print(f"\nTransformer - OOD (N=60):       {acc(transformer, X_test_ood, y_test_ood):.3f}")
print(f"MLP         - OOD: fails (fixed input size, cannot run on N=60)")