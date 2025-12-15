"""
Transfer Learning Model

Pre-trained models (ResNet, VGG, EfficientNet) fine-tuned for plant disease classification.
"""

import logging
import time
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.models as models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferLearningModel:
    """
    Transfer learning wrapper for pre-trained models.
    
    Supported models:
    - ResNet18, ResNet34, ResNet50
    - VGG16, VGG19
    - EfficientNet-B0, B1, B2
    - MobileNetV2
    """
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 num_classes: int = 34,  # Updated for Pakistan dataset (was 39 for PlantVillage)
                 pretrained: bool = True,
                 freeze_layers: bool = True):
        """
        Initialize transfer learning model.
        
        Args:
            model_name (str): Name of pre-trained model
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet pre-trained weights
            freeze_layers (bool): Freeze early layers
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        
        self.model = self._build_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.optimizer = None
        self.criterion = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"TransferLearningModel initialized: {model_name}, "
                   f"pretrained={pretrained}, freeze={freeze_layers}")
    
    def _build_model(self) -> nn.Module:
        """Build transfer learning model."""
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        elif self.model_name == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_classes)
        
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=self.pretrained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Freeze layers if specified
        if self.freeze_layers:
            self._freeze_early_layers(model)
        
        return model
    
    def _freeze_early_layers(self, model: nn.Module) -> None:
        """Freeze early layers for fine-tuning."""
        if 'resnet' in self.model_name:
            # Freeze all except last 2 blocks and fc
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        
        elif 'vgg' in self.model_name:
            # Freeze features, train classifier
            for param in model.features.parameters():
                param.requires_grad = False
        
        elif 'mobilenet' in self.model_name:
            # Freeze early layers
            for name, param in model.named_parameters():
                if 'features.18' not in name and 'classifier' not in name:
                    param.requires_grad = False
        
        logger.info("Early layers frozen for fine-tuning")
    
    def unfreeze_all_layers(self) -> None:
        """Unfreeze all layers for full training."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")
    
    def setup_training(self, 
                      learning_rate: float = 0.001,
                      weight_decay: float = 1e-4):
        """Setup optimizer and loss."""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(train_loader), 100.0 * correct / total
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(val_loader), 100.0 * correct / total
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 30) -> Dict:
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            best_val_acc = max(best_val_acc, val_acc)
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'best_val_acc': best_val_acc,
            'history': self.training_history
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test data."""
        from sklearn.metrics import precision_recall_fscore_support
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TransferLearningModel':
        """Load model."""
        checkpoint = torch.load(filepath)
        model = cls(
            model_name=checkpoint['model_name'],
            num_classes=checkpoint['num_classes'],
            pretrained=False
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        return model
