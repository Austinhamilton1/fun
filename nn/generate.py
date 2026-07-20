import argparse
import torch
import numpy as np
from PIL import Image
from generator import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Generator'
    )

    parser.add_argument('-m', '--model')

    args = parser.parse_args()

    if args.model:
        generator = Generator()

        generator.load_state_dict(
            torch.load(
                f'./models/{args.model}'
            )
        )

        generator.eval()

        noise = torch.randn(
            1,
            1,
            256,
            256,
        )

        with torch.no_grad():
            img = torch.sigmoid(generator(noise))
            img = img.squeeze().cpu().numpy()
            img = (img * 255).astype(np.uint8)

            Image.fromarray(
                img,
                mode='L',
            ).show()
