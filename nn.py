import numpy as np
import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import os
from PIL import Image

class NoiseDataset(Dataset):
    '''Noise loading Dataset'''
    def __init__(self, noise_dir, transform=None):
        self.root = noise_dir
        self.files : list[str] = os.listdir(noise_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert('L')
        sample = np.asarray(img, dtype=np.uint8)

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    '''Normalize the data in a noise sample'''
    def __call__(self, sample):
        return sample.astype(np.float32) / 255.0

class ToTensor(object):
    '''Convert the data in a noise sample to a tensor'''
    def __call__(self, sample):
        return torch.from_numpy(sample)

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

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 400 -> 200
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 200 -> 100
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)               # 100 -> 50
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),                            # 50 -> 100
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16,
                kernel_size=2,
                stride=2
            ),                            # 100 -> 200
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1,
                kernel_size=2,
                stride=2
            ),                            # 200 -> 400
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out.squeeze(1)

    def encode(self, x):
        x = x.unsqueeze(x)
        return self.encoder(x)

class ImageLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()

        self.loss_fn = loss_fn

    def forward(self, logits, input_img):
        return self.loss_fn(logits, input_img)

def train_encoder(model_name, noise_dir, epochs=50):
    composed = Compose([
        Normalize(),
        ToTensor(),
    ])

    data = NoiseDataset(noise_dir, transform=composed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
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

            loss = criterion(batch, output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f'Epoch: {i}, Loss: {loss.item()}')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}.pt')

def train_generator(model_name, noise_dir, loss_fn, epochs=50):
    composed = Compose([
        Normalize(),
        ToTensor(),
    ])

    data = NoiseDataset(noise_dir, transform=composed)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f'Epoch: {i}, Loss: {loss.item()}')

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}.pt')
