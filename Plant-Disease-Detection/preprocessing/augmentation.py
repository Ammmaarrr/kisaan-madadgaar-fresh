"""
Data Augmentation Module

This module provides various image augmentation techniques
for improving model robustness and generalization.
"""

import numpy as np
from typing import Tuple, Optional
import logging

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available")

try:
    import torch
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Data augmentation strategies for plant disease images.
    
    Techniques:
    - Geometric: Rotation, flipping, cropping, scaling
    - Color: Brightness, contrast, saturation, hue adjustment
    - Noise: Gaussian noise, blur
    - Advanced: Mixup, Cutout
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize augmentation module.
        
        Args:
            config (dict): Augmentation configuration
                - rotation_range: Max rotation degrees (default: 15)
                - flip_probability: Horizontal flip prob (default: 0.5)
                - brightness_range: Brightness adjustment (default: 0.2)
                - contrast_range: Contrast adjustment (default: 0.2)
                - saturation_range: Saturation adjustment (default: 0.2)
                - hue_range: Hue adjustment (default: 0.1)
                - zoom_range: Random crop scale (default: (0.8, 1.0))
        """
        self.config = config or {}
        
        self.rotation_range = self.config.get('rotation_range', 15)
        self.flip_probability = self.config.get('flip_probability', 0.5)
        self.brightness_range = self.config.get('brightness_range', 0.2)
        self.contrast_range = self.config.get('contrast_range', 0.2)
        self.saturation_range = self.config.get('saturation_range', 0.2)
        self.hue_range = self.config.get('hue_range', 0.1)
        self.zoom_range = self.config.get('zoom_range', (0.8, 1.0))
        
        logger.info("DataAugmentation initialized")
    
    def get_training_transforms(self, img_size: int = 224):
        """
        Get comprehensive training augmentation pipeline.
        
        Args:
            img_size (int): Target image size
            
        Returns:
            torchvision.transforms.Compose: Transform pipeline
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for transforms")
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=self.flip_probability),
            transforms.RandomRotation(degrees=self.rotation_range),
            transforms.ColorJitter(
                brightness=self.brightness_range,
                contrast=self.contrast_range,
                saturation=self.saturation_range,
                hue=self.hue_range
            ),
            transforms.RandomResizedCrop(
                img_size, 
                scale=self.zoom_range,
                ratio=(0.9, 1.1)
            ),
            # Additional augmentations
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_validation_transforms(self, img_size: int = 224):
        """
        Get validation/test transforms (no augmentation).
        
        Args:
            img_size (int): Target image size
            
        Returns:
            torchvision.transforms.Compose: Transform pipeline
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for transforms")
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_heavy_augmentation(self, img_size: int = 224):
        """
        Get heavy augmentation for small datasets.
        
        Args:
            img_size (int): Target image size
            
        Returns:
            torchvision.transforms.Compose: Heavy augmentation pipeline
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for transforms")
        
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.2
            ),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2)
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10
                )
            ], p=0.4),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def mixup_data(x: torch.Tensor, y: torch.Tensor, 
                   alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Mixup augmentation: mix two images and their labels.
        
        Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
        
        Args:
            x (torch.Tensor): Batch of images
            y (torch.Tensor): Batch of labels
            alpha (float): Mixup hyperparameter
            
        Returns:
            tuple: (mixed_x, y_a, y_b, lambda)
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for mixup")
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def cutout(image: torch.Tensor, n_holes: int = 1, 
               length: int = 16) -> torch.Tensor:
        """
        Cutout augmentation: randomly mask out square regions.
        
        Reference: "Improved Regularization of Convolutional Neural Networks 
                    with Cutout" (DeVries & Taylor, 2017)
        
        Args:
            image (torch.Tensor): Image tensor (C, H, W)
            n_holes (int): Number of holes to cut
            length (int): Length of the square hole
            
        Returns:
            torch.Tensor: Image with cutout applied
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for cutout")
        
        h = image.size(1)
        w = image.size(2)
        
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = mask.unsqueeze(0).expand_as(image)
        image = image * mask
        
        return image
    
    @staticmethod
    def random_crop_with_disease_focus(image: Image.Image, 
                                      crop_size: Tuple[int, int]) -> Image.Image:
        """
        Crop image focusing on diseased areas (center-biased).
        
        Args:
            image (PIL.Image): Input image
            crop_size (tuple): (width, height) of crop
            
        Returns:
            PIL.Image: Cropped image
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL required for cropping")
        
        width, height = image.size
        crop_width, crop_height = crop_size
        
        # Center-biased random crop (diseases often in center)
        center_x = width // 2
        center_y = height // 2
        
        # Sample from Gaussian distribution centered at image center
        left = int(np.clip(
            np.random.normal(center_x - crop_width // 2, width // 6),
            0, width - crop_width
        ))
        top = int(np.clip(
            np.random.normal(center_y - crop_height // 2, height // 6),
            0, height - crop_height
        ))
        
        return image.crop((left, top, left + crop_width, top + crop_height))
    
    def augment_batch(self, images: torch.Tensor, 
                     technique: str = 'standard') -> torch.Tensor:
        """
        Apply augmentation to a batch of images.
        
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W)
            technique (str): 'standard', 'cutout', or 'both'
            
        Returns:
            torch.Tensor: Augmented batch
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        if technique == 'cutout':
            return torch.stack([self.cutout(img) for img in images])
        elif technique == 'both':
            # Apply cutout to half the batch
            batch_size = images.size(0)
            cutout_indices = np.random.choice(
                batch_size, batch_size // 2, replace=False
            )
            for idx in cutout_indices:
                images[idx] = self.cutout(images[idx])
        
        return images
    
    @staticmethod
    def test_time_augmentation(model, image: torch.Tensor, 
                              n_augmentations: int = 5) -> torch.Tensor:
        """
        Test-time augmentation: average predictions over augmented versions.
        
        Args:
            model: Trained model
            image (torch.Tensor): Single image (C, H, W)
            n_augmentations (int): Number of augmented versions
            
        Returns:
            torch.Tensor: Averaged predictions
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original image
            pred = model(image.unsqueeze(0))
            predictions.append(pred)
            
            # Augmented versions
            for _ in range(n_augmentations - 1):
                # Random horizontal flip
                aug_image = image.clone()
                if np.random.random() > 0.5:
                    aug_image = TF.hflip(aug_image)
                
                # Random rotation
                angle = np.random.randint(-10, 10)
                aug_image = TF.rotate(aug_image, angle)
                
                pred = model(aug_image.unsqueeze(0))
                predictions.append(pred)
        
        # Average predictions
        avg_predictions = torch.mean(torch.stack(predictions), dim=0)
        return avg_predictions
    
    def get_augmentation_config(self) -> dict:
        """Get current augmentation configuration."""
        return {
            'rotation_range': self.rotation_range,
            'flip_probability': self.flip_probability,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'saturation_range': self.saturation_range,
            'hue_range': self.hue_range,
            'zoom_range': self.zoom_range
        }
