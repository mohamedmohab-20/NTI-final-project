import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class OrdinalMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

class OrdinalPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        probs_raw = torch.sigmoid(logits)
        p1 = probs_raw[:, 0:1]
        p2 = probs_raw[:, 1:2]
        p_micro = torch.maximum(p1, p2)
        p_macro = p2
        p_normal = 1.0 - p_micro
        p_micro_cls = p_micro - p_macro
        probs = torch.cat([p_normal, p_micro_cls, p_macro], dim=1)
        probs = torch.clamp(probs, 0.0, 1.0)
        row_sums = torch.sum(probs, dim=1, keepdim=True)
        return probs / (row_sums + 1e-7)

def make_ordinal_targets(y_values):
    y_arr = np.asarray(y_values)
    return np.column_stack([
        (y_arr >= 1).astype(np.float32),
        (y_arr >= 2).astype(np.float32)
    ])

def compute_ordinal_pos_weights(y_ordinal):
    pos_counts = np.sum(y_ordinal, axis=0)
    neg_counts = len(y_ordinal) - pos_counts
    weights = neg_counts / np.clip(pos_counts, 1, None)
    return torch.tensor(weights, dtype=torch.float32)

def train_ordinal_model(model, train_loader, val_loader, pos_weights, epochs=120, lr=0.001, patience=15, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pos_weights = pos_weights.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_weights = None
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(bx)
            n_train += len(bx)

        train_loss = total_train_loss / n_train
        history['train_loss'].append(train_loss)

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                vlogits = model(vx)
                vloss = criterion(vlogits, vy)
                total_val_loss += vloss.item() * len(vx)
                n_val += len(vx)

        val_loss = total_val_loss / n_val
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history

def predict_ordinal(model, X_numpy, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    with torch.no_grad():
        tx = torch.tensor(np.asarray(X_numpy, dtype=np.float32)).to(device)
        logits = model(tx)
        probs_raw = torch.sigmoid(logits).cpu().numpy()

    p_macro = probs_raw[:, 1]
    p_micro = np.maximum(probs_raw[:, 0], p_macro)
    p_normal = 1.0 - p_micro
    p_micro_cls = p_micro - p_macro

    probs = np.column_stack([p_normal, p_micro_cls, p_macro])
    probs = np.clip(probs, 0.0, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / (row_sums + 1e-7)
    preds = np.argmax(probs, axis=1)
    return preds, probs

def export_to_onnx(model, input_dim, filepath="kidney_model.onnx"):
    model = model.cpu()
    model.eval()
    predictor = OrdinalPredictor(model).cpu()
    predictor.eval()

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32).cpu()
    torch.onnx.export(
        predictor,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['probabilities'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'probabilities': {0: 'batch_size'}
        }
    )

    try:
        import onnx
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"ONNX check warning: {e}")
    return filepath
