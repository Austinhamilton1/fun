from nn import train_generator, Generator, AutoEncoder
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

def loss(logits):
    raise NotImplemented()

encoder_name = 'encoder'
model_name = 'generator'

if __name__ == '__main__':
    if not os.path.exists(f'models/{encoder_name}.pt'):
        raise RuntimeError('Could not find encoder')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Generator().to(device)
    encoder.load_state_dict(torch.load(f'models/{encoder_name}.pt', map_location=device))

    if not os.path.exists(f'models/{model_name}.pt'):
        train_generator(model_name, 'noise_samples', loss, epochs=50)

    model = Generator().to(device)
    model.load_state_dict(torch.load(f'models/{model_name}.pt', map_location=device))

    img = np.random.randint(0, 256, size=(400, 400), dtype=np.uint8)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    model.eval()

    with torch.no_grad():
        output = model(img)
        output = output.squeeze(0).cpu().numpy()
        output = (output * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(output, mode='L')
        img.show()
