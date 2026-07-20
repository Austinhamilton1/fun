import torch
from torchvision import transforms, datasets
from autoencoder import load_model, sample_scales
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('./data/city', transform=transform)

img, _ = dataset[12]

img = img.unsqueeze(0).to(device)

model = load_model('./models/city_encoder.pt')

reconstruction, latents, fused = model(img, sample_scales())
reconstruction = reconstruction.squeeze(0)
reconstruction = reconstruction.squeeze(0)

data = reconstruction.detach().cpu().numpy()
data = np.clip(data, 0, 1)

image = Image.fromarray((data * 255).astype(np.uint8), mode='L')

image.show()