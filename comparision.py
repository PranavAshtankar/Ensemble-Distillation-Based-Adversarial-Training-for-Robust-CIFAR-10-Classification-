# ============================================================
# EDAAT ROBUSTNESS BENCHMARK (CPU OPTIMIZED RESEARCH VERSION)
# CIFAR-10 | Larger Data | More Epochs | Time Balanced
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= DEVICE =================
DEVICE = torch.device("cpu")
print("Running on:", DEVICE)

# ================= SAVE DIR =================
SAVE_DIR = "./EDAAT_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= CONFIG =================
DATA_ROOT = "./data"
BATCH_SIZE = 128
EPOCHS_BASE = 8
EPOCHS_EDAAT = 8

EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 1   # reduced for CPU efficiency

ALPHA = 0.4
BETA  = 0.3
GAMMA = 0.3

# ================= TRANSFORMS =================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2470,0.2435,0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2470,0.2435,0.2616))
])

# ================= SUBSET =================
def make_subset(dataset, per_class):
    targets = np.array(dataset.targets)
    indices = []
    for c in range(10):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        indices.extend(idx[:per_class])
    return Subset(dataset, indices)

train_full = torchvision.datasets.CIFAR10(
    DATA_ROOT, train=True, download=True, transform=transform_train
)
test_full = torchvision.datasets.CIFAR10(
    DATA_ROOT, train=False, download=True, transform=transform_test
)


# 600 per class → 6000 total
trainset = make_subset(train_full, 600)
# 200 per class → 2000 total
testset  = make_subset(test_full, 200)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=0
)

print("Train size:", len(trainset))
print("Test size :", len(testset))

# ================= ATTACKS =================
def fgsm_attack(model, images, labels):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv = images + EPSILON * images.grad.sign()
    return adv.detach()

def pgd_attack(model, images, labels, steps=PGD_STEPS):
    ori = images.clone().detach()
    images = images + torch.empty_like(images).uniform_(-EPSILON, EPSILON)

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv = images + PGD_ALPHA * images.grad.sign()
        eta = torch.clamp(adv - ori, -EPSILON, EPSILON)
        images = (ori + eta).detach()

    return images

# ================= MODELS =================
model_names = [
    "No Defense",
    "Preprocessing",
    "Gradient Masking",
    "Detection",
    "FGSM Adv Training",
    "EDAAT"
]

models = {name: resnet18(num_classes=10).to(DEVICE) for name in model_names}

# ================= OPTIMIZER =================
def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)

opts = {name: make_optimizer(models[name]) for name in model_names}

schedulers = {
    name: torch.optim.lr_scheduler.StepLR(opts[name], step_size=4, gamma=0.1)
    for name in model_names
}

criterion = nn.CrossEntropyLoss()

# ============================================================
# PHASE 1: TRAIN BASELINES
# ============================================================

print("\n=== Phase 1: Training Baselines ===")

for epoch in range(EPOCHS_BASE):
    print(f"\n[Baseline Epoch {epoch+1}/{EPOCHS_BASE}]")

    for name in model_names[:-1]:
        models[name].train()

    for x, y in trainloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # No Defense
        opts["No Defense"].zero_grad()
        criterion(models["No Defense"](x), y).backward()
        opts["No Defense"].step()

        # Preprocessing
        x_blur = F.avg_pool2d(x, 2)
        opts["Preprocessing"].zero_grad()
        criterion(models["Preprocessing"](x_blur), y).backward()
        opts["Preprocessing"].step()

        # Gradient Masking
        x_noise = x + 0.01 * torch.randn_like(x)
        opts["Gradient Masking"].zero_grad()
        criterion(models["Gradient Masking"](x_noise), y).backward()
        opts["Gradient Masking"].step()

        # Detection
        opts["Detection"].zero_grad()
        criterion(models["Detection"](x), y).backward()
        opts["Detection"].step()

        # FGSM Adv Training
        x_fgsm = fgsm_attack(models["FGSM Adv Training"], x, y)
        opts["FGSM Adv Training"].zero_grad()
        criterion(models["FGSM Adv Training"](x_fgsm), y).backward()
        opts["FGSM Adv Training"].step()

    for name in model_names[:-1]:
        schedulers[name].step()

# ============================================================
# PHASE 2: TRAIN EDAAT
# ============================================================

print("\n=== Phase 2: Training EDAAT ===")

teachers = [
    models["FGSM Adv Training"],
    models["Detection"]
]

for t in teachers:
    t.eval()

for epoch in range(EPOCHS_EDAAT):
    print(f"\n[EDAAT Epoch {epoch+1}/{EPOCHS_EDAAT}]")
    models["EDAAT"].train()

    for x, y in trainloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        x_adv = pgd_attack(models["EDAAT"], x, y)

        opts["EDAAT"].zero_grad()

        out_clean = models["EDAAT"](x)
        out_adv   = models["EDAAT"](x_adv)

        with torch.no_grad():
            teacher_logits = sum(t(x) for t in teachers) / len(teachers)

        loss_clean = criterion(out_clean, y)
        loss_adv   = criterion(out_adv, y)
        loss_dist  = F.kl_div(
            F.log_softmax(out_clean, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean"
        )

        loss = ALPHA*loss_clean + BETA*loss_adv + GAMMA*loss_dist
        loss.backward()
        opts["EDAAT"].step()

    schedulers["EDAAT"].step()
