"""
Simple CNN Baseline Model

A lightweight convolutional neural network baseline for plant disease classification.
This serves as a simple deep learning baseline to compare with more complex models.
"""

import logging
import time
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for plant disease classification.
    
    Architecture:
    - 3 Convolutional blocks (Conv -> ReLU -> MaxPool)
    - 2 Fully connected layers
    - Dropout for regularization
    
    Total parameters: ~5-10M (lightweight)
    """
    
    def __init__(self, num_classes: int = 34, dropout_rate: float = 0.5):  # Updated for Pakistan dataset (was 39 for PlantVillage)
        """
        Initialize Simple CNN.
        
        Args:
            num_classes (int): Number of disease classes (34 for Pakistan dataset)
            dropout_rate (float): Dropout probability
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        
        # Convolutional Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        
        # Convolutional Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        
        # Fully connected layers
        # Feature map size: 128 * 28 * 28 = 100,352
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
        logger.info(f"SimpleCNN initialized: num_classes={num_classes}, "
                   f"dropout={dropout_rate}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input batch (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Class logits (B, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNNTrainer:
    """Trainer class for SimpleCNN model."""
    
    def __init__(self, model: SimpleCNN, device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model (SimpleCNN): Model to train
            device (str, optional): Device ('cuda' or 'cpu')
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"SimpleCNNTrainer initialized on device: {self.device}")
    
    def setup_training(self, 
                      learning_rate: float = 0.001,
                      weight_decay: float = 1e-4,
                      use_scheduler: bool = True):
        """
        Setup optimizer, loss, and scheduler.
        
        Args:
            learning_rate (float): Initial learning rate
            weight_decay (float): L2 regularization
            use_scheduler (bool): Use learning rate scheduler
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        logger.info(f"Training setup: lr={learning_rate}, "
                   f"weight_decay={weight_decay}")
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
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
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              early_stopping_patience: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            dict: Training history and statistics
        """
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'best_val_acc': best_val_acc,
            'final_train_acc': self.training_history['train_acc'][-1],
            'history': self.training_history,
            'epochs_trained': len(self.training_history['train_loss'])
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            dict: Test metrics
        """
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
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        accuracy = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        
        return {
            'accuracy': accuracy / 100.0,  # Convert to 0-1 range
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'num_classes': self.model.num_classes
        }, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        logger.info(f"Checkpoint loaded from {filepath}")
