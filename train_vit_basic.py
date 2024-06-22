from torch.cuda.amp import autocast, GradScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from tqdm import tqdm
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="inat100k",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "vit_small",
    "dataset": "inat_train_baseline",
    "epochs": 100,
    "batch_size": 256,
    }
)

# Parameters
#ROOT_DIR = "/home/aoneill/data/iNat100k"
ROOT_DIR = '/shared/projects/autoarborist/inflated/los_angeles/los_angeles/aerial'
BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 1000
LEARNING_RATE = 0.0001
NUM_WORKERS = 4
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Function to get all unique class names from training and validation directories
def get_all_classes(*dirs):
    all_classes = set()
    for root_dir in dirs:
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                all_classes.add(class_name)
    return sorted(all_classes)

# Get all unique class names from both train and val directories
train_dir = os.path.join(ROOT_DIR, 'train')
val_dir = os.path.join(ROOT_DIR, 'val')
test_dir = os.path.join(ROOT_DIR, 'test')
all_classes = get_all_classes(train_dir, val_dir)

# Create a universal class_to_idx mapping
class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}

# Define transformations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
val_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

##Dataset setup
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_old_classes = train_dataset.classes
train_dataset.class_to_idx = class_to_idx
train_dataset.classes = all_classes
train_dataset.samples = [
                (path, train_dataset.class_to_idx[
                        train_old_classes[target]
                    ]) for path, target in train_dataset.samples]
train_dataset.targets = [s[1] for s in train_dataset.samples]

val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_old_classes = val_dataset.classes
val_dataset.class_to_idx = class_to_idx
val_dataset.classes = all_classes
val_dataset.samples = [
                (path, val_dataset.class_to_idx[
                        val_old_classes[target]
                    ]) for path, target in val_dataset.samples]
val_dataset.targets = [s[1] for s in val_dataset.samples]

test_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
test_old_classes = val_dataset.classes
test_dataset.class_to_idx = class_to_idx
test_dataset.classes = all_classes
test_dataset.samples = [
                (path, test_dataset.class_to_idx[
                        val_old_classes[target]
                    ]) for path, target in test_dataset.samples]
test_dataset.targets = [s[1] for s in test_dataset.samples]
    
assert train_dataset.class_to_idx == class_to_idx
assert val_dataset.class_to_idx == class_to_idx 
# DataLoader setup
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )   
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )   

# Model setup
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_images = 0, 0, 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Scale the loss, perform a backward pass, and scale down gradients in the optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  # Update the scale for next iteration

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_images += targets.size(0)
    print(f'Epoch done. Loss: {total_loss / len(train_loader):.4f}, Acc: {total_correct / total_images:.4f}')
    wandb.log({"train_acc": total_correct / total_images, "train_loss": total_loss / len(train_loader)})

import torch
import torch.nn.functional as F
import numpy as np

def validate_model(model, val_loader, device, num_classes):
    model.eval()
    total_correct_top1, total_correct_top5, total_images = 0, 0, 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Top-1 accuracy
            total_correct_top1 += predicted.eq(targets).sum().item()
            
            # Update class correct and total counts
            for i in range(targets.size(0)):
                class_total[targets[i]] += 1
                if predicted[i] == targets[i]:
                    class_correct[targets[i]] += 1
            
            # Top-5 accuracy
            top5 = torch.topk(outputs, 5, dim=1)[1]
            total_correct_top5 += sum([targets[i] in top5[i] for i in range(targets.size(0))])
            
            total_images += targets.size(0)

    top1_acc = total_correct_top1 / total_images
    top5_acc = total_correct_top5 / total_images
    class_avg_acc = (class_correct / class_total).mean()
    
    print(f'Validation Acc: Top-1: {top1_acc:.4f}, Top-5: {top5_acc:.4f}, Class Avg: {class_avg_acc:.4f}')
    wandb.log({"val_acc_top1": top1_acc, "val_acc_top5": top5_acc, "val_class_avg_acc": class_avg_acc})

def test_model(model, test_loader, device, num_classes):
    model.eval()
    total_correct_top1, total_correct_top5, total_images = 0, 0, 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Top-1 accuracy
            total_correct_top1 += predicted.eq(targets).sum().item()
            
            # Update class correct and total counts
            for i in range(targets.size(0)):
                class_total[targets[i]] += 1
                if predicted[i] == targets[i]:
                    class_correct[targets[i]] += 1
            
            # Top-5 accuracy
            top5 = torch.topk(outputs, 5, dim=1)[1]
            total_correct_top5 += sum([targets[i] in top5[i] for i in range(targets.size(0))])
            
            total_images += targets.size(0)

    top1_acc = total_correct_top1 / total_images
    top5_acc = total_correct_top5 / total_images
    class_avg_acc = (class_correct / class_total).mean()
    
    print(f'Test Acc: Top-1: {top1_acc:.4f}, Top-5: {top5_acc:.4f}, Class Avg: {class_avg_acc:.4f}')
    wandb.log({"test_acc_top1": top1_acc, "test_acc_top5": top5_acc, "test_class_avg_acc": class_avg_acc})

# Main training loop
for epoch in range(NUM_EPOCHS):
    print(f'Starting epoch {epoch + 1}/{NUM_EPOCHS}')
    train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    validate_model(model, val_loader, DEVICE, NUM_CLASSES)

# Evaluate the model on the test set
test_model(model, test_loader, DEVICE, NUM_CLASSES)

# Save the trained model
model_path = os.path.join(ROOT_DIR, 'pretrained_vit_base_patch16_224.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
wandb.finish()
