"""
Train the deepfake image classifier used by pages/Deepfake.py.

IMPORTANT FIX: the current Deepfake.py loads a pretrained ResNet18 and
attaches a brand-new, randomly initialized final layer -- but never
trains or loads trained weights for it. So today it is not actually
detecting deepfakes, it's guessing. This script fine-tunes that final
layer (and optionally more) on a real/fake face dataset and saves
weights that Deepfake.py can load.

Expected folder layout (standard ImageFolder format):
    data/
        train/
            real/*.jpg
            fake/*.jpg
        val/
            real/*.jpg
            fake/*.jpg

Datasets that already come in (or can easily be reshaped into) this layout:
    - Real and Fake Face Detection (Kaggle, ciplab)
    - 140k Real and Fake Faces (Kaggle, xhlulu)

Usage:
    python train_deepfake_model.py --data-dir ./data --epochs 5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


def build_model():
    model = models.resnet18(pretrained=True)
    # Freeze the backbone, train only the new head first (fast, avoids
    # overfitting on smaller datasets). You can unfreeze more layers later.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: real, fake
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder with train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="deepfake_detector.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{args.data_dir}/train", transform=transform)
    val_ds = datasets.ImageFolder(f"{args.data_dir}/val", transform=transform)
    print(f"Classes (index -> label): {train_ds.class_to_idx}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {running_loss/len(train_loader):.4f} "
              f"- val_acc: {val_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_ds.class_to_idx,
    }, args.out)
    print(f"\nSaved trained weights to {args.out}")
    print("Update Deepfake.py to load this checkpoint instead of building an untrained head.")


if __name__ == "__main__":
    main()
