import torch.nn as nn 
import torchsummary

class BrainMRINet(nn.Module):
    def __init__(self, num_classes=2, input_size=(130, 130)):
        super(BrainMRINet, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  
            
            # Block 4 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), 
            
            # Block 5 
            nn.AdaptiveAvgPool2d((1, 1))  # Output: 128 x 1 x 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Output: 128
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    model = BrainMRINet(num_classes=2, input_size=(130, 130))
    print(model.classifier[-1].out_features)
    

