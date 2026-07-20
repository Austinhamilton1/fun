from torch import nn
from torch._dynamo.config import same_two_models_use_fp64
from torchvision.models.squeezenet import torch
from tqdm import tqdm
from autoencoder import load_model, sample_scales, make_scaled_image, scale_consistency_loss

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
        return self.network(x)

def variance_loss(img):
    '''
    Prevent generator collapse

    img:
        (B, 1, H, W)
    '''
    variance = torch.var(img, dim=[1,2,3])

    # Maximize variance
    return -variance.mean()

def generator_scale_loss(
    encoder,
    images,
    num_scales=4,
):
    scales = sample_scales(num_scales)

    latents = []
    for s in scales:
        scaled = make_scaled_image(images, same_two_models_use_fp64)
        z = encoder.encode(scaled)
        latents.append(z)

    latents = torch.stack(
        latents,
        dim=1,
    )

    return scale_consistency_loss(
        latents,
        scales,
    )

def train_generator(
    generator,
    encoder,
    device='cuda',
    epochs=1000,
    batch_size=8,
    lr=1e-4,
    image_size=256,
):
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=lr,
    )

    generator.to(device)

    for epoch in tqdm(range(epochs)):
        generator.train()

        # Random input noise
        noise = torch.randn(
            batch_size,
            1,
            image_size,
            image_size,
            device=device,
        )

        generated = generator(noise)
        generated = torch.sigmoid(generated)

        # Scale equivariance
        latent_loss = generator_scale_loss(
            encoder,
            generated,
        )

        # Prevent collapse
        diversity_loss = variance_loss(generated)

        loss = (
            latent_loss
            + 0.01 * diversity_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return generator

def save_generator(
    filename,
    encoder_path,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = load_model(encoder_path).to(device)

    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    generator = Generator()

    generator = train_generator(
        generator,
        encoder,
        device=device,
    )

    torch.save(
        generator.state_dict(),
        filename,
    )
