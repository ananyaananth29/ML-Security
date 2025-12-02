import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from models.resnet import ResNet18

# ---- PGD Attack Function (from your training scripts) ----
def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    ori_images = images.clone().detach()

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.sign()
        images = images + alpha * grad
        eta = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images

# ---- Load all models ----
checkpoint_paths = {
    "Standard": "checkpoint_basic_training/basic_training",
    "RobustOnly": "checkpoint_basic_training_with_robust_dataset/basic_training_with_robust_dataset",
    "PGDOnly": "checkpoint_pgd_training/pgd_adversarial_training",
    "Robust+PGD": "checkpoint_pgd_and_robust/pgd_training_with_robust_dataset.pth"
}

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 test loader
test_transform = transforms.Compose([transforms.ToTensor()])
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

# ---- Load all models into memory ----
models = {}
for name, path in checkpoint_paths.items():
    model = ResNet18().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['net'])
    model.eval()
    models[name] = model

# ---- Transfer Evaluation ----
results = {source: {} for source in models.keys()}

for source_name, source_model in models.items():
    print(f"\nGenerating adversarial examples from: {source_name}")
    correct_targets = {target_name: 0 for target_name in models.keys()}
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(source_model, images, labels)

        for target_name, target_model in models.items():
            with torch.no_grad():
                outputs = target_model(adv_images)
                _, preds = outputs.max(1)
                correct_targets[target_name] += preds.eq(labels).sum().item()

        total += labels.size(0)

    for target_name in models.keys():
        acc = 100.0 * correct_targets[target_name] / total
        results[source_name][target_name] = acc

# ---- Print Transfer Matrix ----
print("\n=== Transfer Attack Robustness Matrix (Accuracies %) ===")
header = "From\\To\t" + "\t".join(models.keys())
print(header)
for src in models.keys():
    row = f"{src}\t" + "\t".join(f"{results[src][tgt]:.2f}" for tgt in models.keys())
    print(row)
