"""
Pakistan Plant Disease Detection - Local Training Script
Uses local data at D:\\kisaan madadgaar\\Plant-Disease-Detection\\data
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import json
from collections import defaultdict

# ============================================
# 1. DATA DIRECTORY SETUP
# ============================================
print("=" * 60)
print("STEP 1: DATA DIRECTORY SETUP")
print("=" * 60)

DATA_DIR = Path(r"d:\kisaan madadgaar\Plant-Disease-Detection\data")
print(f"\n📂 Data Directory: {DATA_DIR}")
print(f"✅ Exists: {DATA_DIR.exists()}")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

# List subfolders
DATA_SUBFOLDERS = []
subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
print(f"\n📁 Found {len(subdirs)} subfolder(s):")

for subdir in subdirs:
    file_count = sum(1 for _ in subdir.rglob('*') if _.is_file())
    print(f"   - {subdir.name}: {file_count:,} files")
    DATA_SUBFOLDERS.append(subdir.name)

print(f"\n✅ DATA_SUBFOLDERS = {DATA_SUBFOLDERS}")

# ============================================
# 2. PYTORCH SETUP
# ============================================
print("\n" + "=" * 60)
print("STEP 2: PYTORCH SETUP")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
print(f"📦 PyTorch version: {torch.__version__}")

# Mixed precision training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
print(f"⚡ AMP (Automatic Mixed Precision): {'Enabled' if scaler else 'Disabled'}")

# ============================================
# 3. DATASET ANALYSIS
# ============================================
print("\n" + "=" * 60)
print("STEP 3: ANALYZING DATASET STRUCTURE")
print("=" * 60)

def find_image_classes(root_path, max_depth=5, min_images=50):
    """
    Recursively find image folders and count images
    """
    class_data = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    def scan_directory(path, depth=0):
        if depth > max_depth:
            return
        
        try:
            items = list(path.iterdir())
            image_files = [f for f in items if f.is_file() and f.suffix in image_extensions]
            
            # If this folder has images, record it
            if len(image_files) >= min_images:
                class_data[str(path)] = len(image_files)
            
            # Recursively scan subdirectories
            for item in items:
                if item.is_dir():
                    scan_directory(item, depth + 1)
        except PermissionError:
            pass
    
    scan_directory(Path(root_path))
    return class_data

# Analyze all subfolders
all_classes = {}
for subfolder in DATA_SUBFOLDERS:
    subfolder_path = DATA_DIR / subfolder
    print(f"\n🔍 Analyzing: {subfolder}")
    
    classes_found = find_image_classes(subfolder_path, max_depth=5, min_images=50)
    
    # Add to main dictionary with folder prefix
    for class_path, count in classes_found.items():
        class_name = Path(class_path).name
        key = f"{subfolder}___{class_name}"
        all_classes[key] = {'path': class_path, 'count': count}
    
    print(f"   Found {len(classes_found)} classes with 50+ images")

total_classes = len(all_classes)
total_images = sum(data['count'] for data in all_classes.values())

print(f"\n" + "=" * 60)
print(f"📊 DATASET SUMMARY")
print("=" * 60)
print(f"Total Classes: {total_classes}")
print(f"Total Images: {total_images:,}")
print(f"\nSample Classes:")
for i, (class_name, data) in enumerate(list(all_classes.items())[:5]):
    print(f"   {i+1}. {class_name}: {data['count']} images")

# ============================================
# 4. DATASET CLASS
# ============================================
print("\n" + "=" * 60)
print("STEP 4: CREATING DATASET")
print("=" * 60)

class PlantDiseaseDataset(Dataset):
    def __init__(self, all_classes_dict, transform=None, split='train', train_ratio=0.8):
        """
        Dataset that loads images from all class folders
        """
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Create class index mapping
        idx = 0
        for class_name in sorted(all_classes_dict.keys()):
            self.class_to_idx[class_name] = idx
            idx += 1
        
        # Load images from each class
        for class_name, data in all_classes_dict.items():
            class_path = Path(data['path'])
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            image_files = [f for f in class_path.iterdir() 
                          if f.is_file() and f.suffix in image_extensions]
            
            # Split into train/test
            split_point = int(len(image_files) * train_ratio)
            
            if split == 'train':
                selected_files = image_files[:split_point]
            else:  # test
                selected_files = image_files[split_point:]
            
            for img_file in selected_files:
                self.images.append(str(img_file))
                self.labels.append(class_idx)
        
        print(f"\n✅ {split.upper()} Dataset: {len(self.images):,} images, {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image on error
            return torch.zeros(3, 224, 224), label

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = PlantDiseaseDataset(all_classes, transform=train_transform, split='train', train_ratio=0.8)
test_dataset = PlantDiseaseDataset(all_classes, transform=test_transform, split='test', train_ratio=0.8)

# Create dataloaders
batch_size = 32  # Reduced for CPU
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for Windows
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\n✅ DataLoaders created (batch_size={batch_size})")

# ============================================
# 5. MODEL CREATION
# ============================================
print("\n" + "=" * 60)
print("STEP 5: CREATING MODEL")
print("=" * 60)

num_classes = len(train_dataset.class_to_idx)
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
model = model.to(device)

print(f"\n✅ Model: EfficientNet-B4")
print(f"📊 Classes: {num_classes}")
print(f"🎯 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# ============================================
# 6. TRAINING
# ============================================
print("\n" + "=" * 60)
print("STEP 6: TRAINING")
print("=" * 60)

num_epochs = 2  # Reduced for testing
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_loss = test_running_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    print(f'\nEpoch {epoch+1}:')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

# ============================================
# 7. SAVE MODEL
# ============================================
print("\n" + "=" * 60)
print("STEP 7: SAVING MODEL")
print("=" * 60)

save_path = Path(r"d:\kisaan madadgaar\Plant-Disease-Detection\saved_models")
save_path.mkdir(exist_ok=True)

model_file = save_path / 'pakistan_plant_disease_model.pth'
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_acc': train_accs[-1],
    'test_acc': test_accs[-1],
    'class_to_idx': train_dataset.class_to_idx,
    'num_classes': num_classes
}, model_file)

print(f"\n✅ Model saved: {model_file}")
print(f"📊 Final Test Accuracy: {test_accs[-1]:.2f}%")

# Save training history
history_file = save_path / 'training_history.json'
with open(history_file, 'w') as f:
    json.dump({
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }, f, indent=2)

print(f"✅ History saved: {history_file}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
