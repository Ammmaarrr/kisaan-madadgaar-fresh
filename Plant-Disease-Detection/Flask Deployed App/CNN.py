import pandas as pd
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out


# Updated for Pakistan dataset (34 classes) - includes Rice, Cotton, Wheat, Mango + PlantVillage subset
idx_to_classes = {0: 'Cotton - Diseased Cotton Leaf',
                  1: 'Cotton - Diseased Cotton Plant',
                  2: 'Cotton - Fresh Cotton Leaf',
                  3: 'Cotton - Fresh Cotton Plant',
                  4: 'Mango - Anthracnose',
                  5: 'Mango - Bacterial Canker',
                  6: 'Mango - Cutting Weevil',
                  7: 'Mango - Die Back',
                  8: 'Mango - Gall Midge',
                  9: 'Mango - Healthy',
                  10: 'Mango - Powdery Mildew',
                  11: 'Mango - Sooty Mould',
                  12: 'Pepper (Bell) - Bacterial Spot',
                  13: 'Pepper (Bell) - Healthy',
                  14: 'Potato - Early Blight',
                  15: 'Potato - Late Blight',
                  16: 'Potato - Healthy',
                  17: 'Tomato - Bacterial Spot',
                  18: 'Tomato - Early Blight',
                  19: 'Tomato - Late Blight',
                  20: 'Tomato - Leaf Mold',
                  21: 'Tomato - Septoria Leaf Spot',
                  22: 'Tomato - Spider Mites (Two-spotted Spider Mite)',
                  23: 'Tomato - Target Spot',
                  24: 'Tomato - Yellow Leaf Curl Virus',
                  25: 'Tomato - Mosaic Virus',
                  26: 'Tomato - Healthy',
                  27: 'Rice - Brown Spot',
                  28: 'Rice - Healthy',
                  29: 'Rice - Hispa',
                  30: 'Rice - Leaf Blast',
                  31: 'Wheat - Healthy',
                  32: 'Wheat - Septoria',
                  33: 'Wheat - Stripe Rust'}
