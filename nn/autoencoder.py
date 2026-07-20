import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels=1,
        base_channels=64,
        latent_channels=512,
        image_size=256,
        latent_size=8,
    ):
        super().__init__()

        assert image_size % latent_size == 0

        num_down = int(math.log2(image_size // latent_size))

        channels = base_channels
        
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_down):
            next_channels = min(channels * 2, latent_channels)

            layers.extend([
                nn.Conv2d(
                    channels,
                    next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            ])

            channels = next_channels
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class LatentFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.score = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, latents):
        '''
        latents:
            (B, N, C, H, W)
        '''
        B, N, C, H, W = latents.shape
        
        scores = []
        for i in range(N):
            scores.append(self.score(latents[:, i]))

        scores = torch.stack(scores, dim=1)
        weights = torch.softmax(scores, dim=1)
        fused = (weights * latents).sum(dim=1)

        return fused
    
class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        base_channels=64,
        latent_channels=512,
        image_size=256,
        latent_size=8,
    ):
        super().__init__()

        num_up = int(math.log2(image_size // latent_size))

        channels = latent_channels

        layers = []
        for _ in range(num_up):
            next_channels = max(channels // 2, base_channels)

            layers.extend([
                nn.ConvTranspose2d(
                    channels,
                    next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            ])

            channels = next_channels

        layers.append(
            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        )

        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class ScaleAutoEncoder(nn.Module):
    def __init__(
        self, 
        image_size=256,
        latent_size=8,
        latent_channels=512,
    ):
        super().__init__()

        self.encoder = Encoder(
            image_size=image_size,
            latent_size=latent_size,
            latent_channels=latent_channels,
        )

        self.fusion = LatentFusion(latent_channels)

        self.decoder = Decoder(
            image_size=image_size,
            latent_size=latent_size,
            latent_channels=latent_channels,
        )

    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x, scales):
        latents = []
        for s in scales:
            scaled = make_scaled_image(x, s)
            latents.append(
                self.encoder(scaled)
            )

        latents = torch.stack(latents, dim=1)
        fused = self.fusion(latents)
        reconstruction = self.decoder(fused)

        return reconstruction, latents, fused
    
def sample_scales(num_scales=4):
    scales = []

    while len(scales) < num_scales:
        s = random.uniform(1.0, 4.0)

        # Prevent scales from becoming nearly identical
        if all(abs(s - t) > 0.1 for t in scales):
            scales.append(s)

    return scales

def make_scaled_image(img, scale):
    '''
    img: (B, C, H, W)
    '''
    H, W = img.shape[-2:]

    small_h = max(1, int(H / scale))
    small_w = max(1, int(W / scale))

    x = F.interpolate(
        img,
        size=(small_h, small_w),
        mode='bilinear',
        align_corners=False,
    )

    x = F.interpolate(
        x,
        size=(H, W),
        mode='bilinear',
        align_corners=False,
    )

    return x

def scale_consistency_loss(latents, scales, eps=1e-6):
    '''
    latents:
        (B, N, C, H, W)

    scales:
        python list of floats
    '''
    device = latents.device

    scales = torch.tensor(scales, device=device)

    total_loss = 0.0
    total_weight = 0.0

    N = len(scales)

    for i in range(N):
        for j in range(i+1, N):
            weight = 1.0 / (
                torch.abs(torch.log(scales[i]) - torch.log(scales[j]))
                + eps
            )

            loss = F.mse_loss(
                latents[:, i],
                latents[:, j],
            )

            total_loss += weight * loss
            total_weight += weight

    return total_loss / total_weight

def train_autoencoder(
    train_loader,
    image_size=256,
    latent_size=8,
    latent_channels=256,
    device='cuda',
    epochs=20,
    lr=1e-4,    
):
    model = ScaleAutoEncoder(
        image_size=image_size,
        latent_size=latent_size,
        latent_channels=latent_channels,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    for epoch in range(epochs):
        model.train()

        running_recon = 0.0
        running_latent = 0.0

        pbar = tqdm(train_loader)

        for images, *_ in pbar:
            images = images.to(device)
            scales = sample_scales()

            reconstruction, latents, fused = model(
                images,
                scales,
            )

            recon_loss = F.mse_loss(
                reconstruction,
                images,
            )

            latent_loss = scale_consistency_loss(
                latents,
                scales,
            )

            loss = recon_loss + 0.25 * latent_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_recon += recon_loss.item()
            running_latent += latent_loss.item()

            pbar.set_description(
                f'Epoch {epoch+1}/{epochs}'
            )

            pbar.set_postfix(
                recon=f'{recon_loss.item():.5f}',
                latent=f'{latent_loss.item():.5f}'
            )

        print(
            f'Epoch {epoch+1}: '
            f'Recon={running_recon/len(train_loader):.5f} '
            f'Latent={running_latent/len(train_loader):.5f}'
        )
    
    return model

def save_model(dataset, model_filename, img_size=256):
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = train_autoencoder(
        train_loader=train_loader,
        image_size=img_size,
        latent_size=8,
        latent_channels=256,
        device=device,
        epochs=20,
    )

    torch.save(
        {
            'image_size': img_size,
            'latent_size': 8,
            'latent_channels': 256,
            'state_dict': model.state_dict(),
        },
        model_filename,
    )

def load_model(model_filename):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    checkpoint = torch.load(
        model_filename,
        map_location=device,
    )

    model = ScaleAutoEncoder(
        image_size=checkpoint['image_size'],
        latent_size=checkpoint['latent_size'],
        latent_channels=checkpoint['latent_channels'],
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model