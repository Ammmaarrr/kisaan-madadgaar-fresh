"""
Data Preprocessing Pipeline for Plant Disease Detection

This module implements a comprehensive preprocessing pipeline with:
- Data loading from PlantVillage dataset
- Exploratory Data Analysis (EDA)
- Image preprocessing and normalization
- Data augmentation
- Train/validation/test splitting
- DataLoader creation
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from collections import Counter

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import transforms
    from PIL import Image
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Some features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Visualization features disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDiseaseDataset(Dataset):
    """Custom Dataset for Plant Disease images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, class_names: List[str] = None):
        """
        Initialize dataset.
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of integer labels
            transform: torchvision transforms
            class_names (list): List of class names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or []
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name for a label index."""
        if idx < len(self.class_names):
            return self.class_names[idx]
        return f"Class_{idx}"


class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline for plant disease detection.
    
    Features:
    - Load data from directory structure or CSV
    - Comprehensive EDA with visualizations
    - Image preprocessing (resize, normalize)
    - Data augmentation
    - Train/val/test splitting
    - DataLoader creation with batching
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config (dict): Configuration dictionary with keys:
                - img_size: Image size (default: 224)
                - batch_size: Batch size (default: 32)
                - augmentation: Enable augmentation (default: True)
                - train_ratio: Training set ratio (default: 0.7)
                - val_ratio: Validation set ratio (default: 0.15)
                - test_ratio: Test set ratio (default: 0.15)
                - num_workers: DataLoader workers (default: 4)
        """
        self.config = config or {}
        
        # Set defaults
        self.img_size = self.config.get('img_size', 224)
        self.batch_size = self.config.get('batch_size', 32)
        self.augmentation_enabled = self.config.get('augmentation', True)
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.val_ratio = self.config.get('val_ratio', 0.15)
        self.test_ratio = self.config.get('test_ratio', 0.15)
        self.num_workers = self.config.get('num_workers', 4)
        
        # Data storage
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.dataset_stats = {}
        
        # Transforms
        self.train_transform = None
        self.test_transform = None
        self._setup_transforms()
        
        logger.info(f"DataPreprocessingPipeline initialized: img_size={self.img_size}, "
                   f"batch_size={self.batch_size}, augmentation={self.augmentation_enabled}")
    
    def _setup_transforms(self):
        """Setup image transformations."""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available. Transforms not configured.")
            return
        
        # ImageNet normalization values (standard for pretrained models)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms (with augmentation)
        if self.augmentation_enabled:
            self.train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize
            ])
        
        # Test/validation transforms (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize
        ])
        
        logger.info("Transforms configured successfully")
    
    def load_data(self, data_path: str, structure: str = 'directory') -> None:
        """
        Load data from directory structure or CSV.
        
        Args:
            data_path (str): Path to data directory or CSV file
            structure (str): 'directory' or 'csv'
                - directory: data_path/class_name/image.jpg
                - csv: CSV with columns ['image_path', 'label']
        """
        if structure == 'directory':
            self._load_from_directory(data_path)
        elif structure == 'csv':
            self._load_from_csv(data_path)
        else:
            raise ValueError(f"Unknown structure: {structure}")
        
        logger.info(f"Loaded {len(self.image_paths)} images from {len(self.class_names)} classes")
    
    def _load_from_directory(self, data_path: str) -> None:
        """
        Load data from directory structure.
        
        Expected structure:
        data_path/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Get class directories
        class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load images
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            # Get all images in class directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            for ext in image_extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
        
        logger.info(f"Loaded from directory: {len(self.class_names)} classes, "
                   f"{len(self.image_paths)} images")
    
    def _load_from_csv(self, csv_path: str) -> None:
        """
        Load data from CSV file.
        
        CSV should have columns: ['image_path', 'label']
        """
        df = pd.read_csv(csv_path)
        
        if 'image_path' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'image_path' and 'label' columns")
        
        self.image_paths = df['image_path'].tolist()
        
        # Handle label encoding
        if df['label'].dtype == 'object':
            # String labels - create encoding
            unique_labels = sorted(df['label'].unique())
            self.class_names = unique_labels
            self.class_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
            self.labels = [self.class_to_idx[label] for label in df['label']]
        else:
            # Numeric labels
            self.labels = df['label'].tolist()
            self.class_names = [f"Class_{i}" for i in range(max(self.labels) + 1)]
        
        logger.info(f"Loaded from CSV: {len(self.class_names)} classes, "
                   f"{len(self.image_paths)} images")
    
    def exploratory_data_analysis(self, save_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive EDA.
        
        Args:
            save_dir (str, optional): Directory to save visualizations
            
        Returns:
            dict: EDA statistics
        """
        if not self.image_paths:
            logger.warning("No data loaded. Call load_data() first.")
            return {}
        
        logger.info("Performing Exploratory Data Analysis...")
        
        # Class distribution
        label_counts = Counter(self.labels)
        class_distribution = {
            self.class_names[label]: count 
            for label, count in label_counts.items()
        }
        
        # Image statistics
        sample_images = self._sample_images_for_stats(num_samples=100)
        
        stats = {
            'num_classes': len(self.class_names),
            'num_images': len(self.image_paths),
            'class_names': self.class_names,
            'class_distribution': class_distribution,
            'images_per_class': {
                'mean': np.mean(list(label_counts.values())),
                'std': np.std(list(label_counts.values())),
                'min': min(label_counts.values()),
                'max': max(label_counts.values())
            },
            'image_stats': sample_images
        }
        
        self.dataset_stats = stats
        
        # Create visualizations
        if PLOTTING_AVAILABLE and save_dir:
            self._create_eda_visualizations(stats, save_dir)
        
        # Print summary
        self._print_eda_summary(stats)
        
        return stats
    
    def _sample_images_for_stats(self, num_samples: int = 100) -> Dict:
        """Sample images to calculate statistics."""
        sample_indices = np.random.choice(
            len(self.image_paths), 
            min(num_samples, len(self.image_paths)), 
            replace=False
        )
        
        widths, heights, channels = [], [], []
        
        for idx in sample_indices:
            try:
                img = Image.open(self.image_paths[idx])
                w, h = img.size
                widths.append(w)
                heights.append(h)
                channels.append(len(img.getbands()))
            except Exception as e:
                logger.warning(f"Error loading image {self.image_paths[idx]}: {e}")
        
        return {
            'width': {'mean': np.mean(widths), 'std': np.std(widths)},
            'height': {'mean': np.mean(heights), 'std': np.std(heights)},
            'channels': Counter(channels)
        }
    
    def _create_eda_visualizations(self, stats: Dict, save_dir: str) -> None:
        """Create and save EDA visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Class distribution bar plot
        plt.figure(figsize=(15, 6))
        classes = list(stats['class_distribution'].keys())
        counts = list(stats['class_distribution'].values())
        
        plt.bar(range(len(classes)), counts)
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution')
        plt.xticks(range(len(classes)), classes, rotation=90, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sample images from each class
        self._plot_sample_images(save_dir)
        
        logger.info(f"EDA visualizations saved to {save_dir}")
    
    def _plot_sample_images(self, save_dir: str, samples_per_class: int = 3) -> None:
        """Plot sample images from each class."""
        num_classes = len(self.class_names)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(samples_per_class * 3, num_classes * 3))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get images for this class
            class_image_paths = [
                self.image_paths[i] for i, label in enumerate(self.labels) 
                if label == class_idx
            ]
            
            # Sample images
            sample_paths = np.random.choice(
                class_image_paths, 
                min(samples_per_class, len(class_image_paths)), 
                replace=False
            )
            
            for img_idx, img_path in enumerate(sample_paths):
                try:
                    img = Image.open(img_path)
                    axes[class_idx, img_idx].imshow(img)
                    axes[class_idx, img_idx].axis('off')
                    if img_idx == 0:
                        axes[class_idx, img_idx].set_title(class_name, fontsize=8)
                except Exception as e:
                    logger.warning(f"Error plotting image {img_path}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _print_eda_summary(self, stats: Dict) -> None:
        """Print EDA summary."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS SUMMARY")
        print("="*60)
        print(f"Number of Classes: {stats['num_classes']}")
        print(f"Total Images: {stats['num_images']}")
        print(f"\nImages per Class:")
        print(f"  Mean: {stats['images_per_class']['mean']:.2f}")
        print(f"  Std:  {stats['images_per_class']['std']:.2f}")
        print(f"  Min:  {stats['images_per_class']['min']}")
        print(f"  Max:  {stats['images_per_class']['max']}")
        print(f"\nImage Dimensions:")
        print(f"  Width:  {stats['image_stats']['width']['mean']:.2f} ± "
              f"{stats['image_stats']['width']['std']:.2f}")
        print(f"  Height: {stats['image_stats']['height']['mean']:.2f} ± "
              f"{stats['image_stats']['height']['std']:.2f}")
        print("="*60 + "\n")
    
    def create_dataloaders(self, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.
        
        Args:
            shuffle_train (bool): Whether to shuffle training data
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DataLoader creation")
        
        if not self.image_paths:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Calculate split sizes
        total_size = len(self.image_paths)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Create datasets
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
        # Shuffle indices
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist()
        
        # Create datasets with respective indices
        train_paths = [self.image_paths[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        
        val_paths = [self.image_paths[i] for i in val_indices]
        val_labels = [self.labels[i] for i in val_indices]
        
        test_paths = [self.image_paths[i] for i in test_indices]
        test_labels = [self.labels[i] for i in test_indices]
        
        train_dataset = PlantDiseaseDataset(
            train_paths, train_labels, 
            transform=self.train_transform,
            class_names=self.class_names
        )
        
        val_dataset = PlantDiseaseDataset(
            val_paths, val_labels,
            transform=self.test_transform,
            class_names=self.class_names
        )
        
        test_dataset = PlantDiseaseDataset(
            test_paths, test_labels,
            transform=self.test_transform,
            class_names=self.class_names
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            pin_memory=False  # Disabled to prevent hangs with large datasets
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False  # Disabled to prevent hangs with large datasets
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False  # Disabled to prevent hangs with large datasets
        )
        
        logger.info(f"DataLoaders created: Train={len(train_dataset)}, "
                   f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            torch.Tensor: Class weights
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for class weights")
        
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.class_names)
        
        # Calculate weights: total_samples / (num_classes * class_count)
        weights = []
        for class_idx in range(num_classes):
            class_count = label_counts.get(class_idx, 1)
            weight = total_samples / (num_classes * class_count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def save_statistics(self, filepath: str) -> None:
        """Save dataset statistics to JSON file."""
        import json
        
        # Convert numpy types to native Python types
        stats_serializable = self._make_serializable(self.dataset_stats)
        
        with open(filepath, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        
        logger.info(f"Statistics saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
