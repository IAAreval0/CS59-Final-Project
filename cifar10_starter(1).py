"""
CS559 Final Project — CIFAR-10 Classification
Classical vs Deep Comparison

Author: Anthony Alarcon

This provides scaffolding for data loading, model definitions, and basic training/evaluation loops.
Expand with full feature extraction, hyperparameter sweeps, visualizations, and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# For deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_cifar10(data_root="./data", val_split=0.1, seed=42):
    """Download CIFAR-10 and create train/val/test loaders or arrays."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
    ])

    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # Simple train/val split
    train_size = int((1 - val_split) * len(train_full))
    val_size = len(train_full) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_full, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    return train_set, val_set, test_set, train_full.classes

# Example usage:
# train_set, val_set, test_set, classes = load_cifar10()
# print(f"Classes: {classes}")
# print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# ---------------------------------------------------------------------------
# 2. Classical Baseline Skeleton (HOG + Linear SVM)
# ---------------------------------------------------------------------------

def extract_hog_features(images, pixels_per_cell=(8, 8), orientations=9):
    """
    Placeholder: extract HOG from a batch of images.
    Group should implement properly with skimage or OpenCV and vectorize.
    """
    from skimage.feature import hog
    from skimage.color import rgb2gray

    features = []
    for img in images:
        # img expected as (C, H, W) tensor or numpy
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        gray = rgb2gray(img)
        fd = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=(2, 2), visualize=False, feature_vector=True)
        features.append(fd)
    return np.array(features)

def train_classical_baseline(X_train, y_train, X_val, y_val):
    """Train a Linear SVM on extracted features. Expand with grid search etc."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
    clf.fit(X_train_s, y_train)

    val_pred = clf.predict(X_val_s)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy (Linear SVM): {acc:.4f}")
    return clf, scaler, acc

# ---------------------------------------------------------------------------
# 3. Deep Learning Skeleton (Simple CNN or ResNet)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Minimal CNN for CIFAR-10 — group should improve architecture."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_resnet18(num_classes=10, pretrained=True):
    """Transfer learning baseline."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels

# ---------------------------------------------------------------------------
# 4. Main experiment driver (skeleton)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CS559 Project — run this to verify data download and basic setup.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_set, val_set, test_set, classes = load_cifar10()
    print(f"Loaded CIFAR-10. Classes: {classes}")
    print(f"Sizes — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Example: create loaders
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

    # Quick deep model sanity check
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- Running 1 epoch sanity check (SimpleCNN) ---")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("\nSetup complete. Next steps:")
    print("- Implement proper HOG / other classical features and train LinearSVC")
    print("- Train the CNN / ResNet for more epochs and plot curves")
    print("- Add evaluation metrics, confusion matrices, and failure analysis")
    print("- Expand the report based on the proposal document")
