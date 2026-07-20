import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # Lift image into feature space
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            # Back to grayscale
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.network(x)
        return x.squeeze(1)

class ImageLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()

        self.loss_fn = loss_fn

    def forward(self, logits, input_img):
        return self.loss_fn(logits, input_img)

def train_generator(model_name, noise_dir, loss_fn, epochs=50):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    data = datasets(root=noise_dir, transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Generator().to(device)
    criterion = ImageLoss(loss_fn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    loader = DataLoader(
        data,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model.train()
    for i in range(epochs):
        for j, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch)

            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {i}, Loss: {loss.item()}')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}.pt')
