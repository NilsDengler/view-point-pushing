import torch 
import torch.nn as nn
import numpy as np
class push_predictions(nn.Module):
    def __init__(self, width, height):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        test = np.ones((32, 1, width, height))
        with torch.no_grad():
            n_features = self.encoder(
                torch.as_tensor(test).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(in_features=n_features, out_features=1))

    def forward(self, x):
        x = self.encoder(x)
        return self.linear(x)
    

if __name__ == "__main__":
    x = torch.rand(32, 1, 400, 400)
    model = push_predictions(400, 400)
    print(model(x).shape)