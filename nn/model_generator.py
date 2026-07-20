import argparse
import requests
from pathlib import Path
import time
import os
import random

# Wikimedia commons requires a unique User-Agent header
# They prefer that the header includes contact information
HEADERS = {
    'User-Agent': 'CNNTrainerBot/1.0 (austinhamilton034@gmail.com)',
}

session = requests.Session()
session.headers.update(HEADERS)

def fetch_image_urls(query, count=50):
    '''
    Fetches direct file links using API pagination, 50 at a time
    '''
    # Wikimedia commons API (freely available images)
    url = 'https://commons.wikimedia.org/w/api.php'
    image_urls = []
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'search',  # Use search as a generator for titles
        'gsrsearch': f'File:{query}',
        'gsrnamespace': 6,      # Namespace 6 targets 'File:' uploads exclusively
        'gsrlimit': 50,
        'prop': 'imageinfo',    # Request information about the files
        'iiprop': 'url',        # Specific metadata properties to pull
        'iiurlwidth': 512,
    }

    while len(image_urls) < count:
        print(f'Fetching metadata... (Gathered {len(image_urls)}/{count})')
        response = session.get(url, params=params)

        if response.status_code != 200:
            print(f'Error reading API: {response.status_code}')
            break

        data = response.json()

        # Parse the page results
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if 'imageinfo' in page_data:
                direct_url = page_data['imageinfo'][0]['url']
                image_urls.append(direct_url)
                if len(image_urls) >= count:
                    break

        # Handle API pagination
        if 'continue' in data:
            params.update(data['continue'])
            # Small courtesy sleep between API metadata pagination requests
            time.sleep(0.5)
        else:
            print('No more search results available')
            break

    return image_urls[:count]

def download_images(urls, folder):
    '''
    Donwload files sequentially to avoid rate-limiting triggers
    '''
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    for idx, url in enumerate(urls, start=1):
        filename = os.path.basename(url)
        filename = filename.split('?')[0]

        save_path = folder / f'{idx}_{filename}'

        try:
            print(f'Downloading [{idx}/{len(urls)}]: {filename}')
            res = session.get(url, stream=True)

            if res.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
            elif res.status_code == 429:
                retry = int(res.headers.get('Retry-After', '30'))
                print(f'Hit rate limit! Sleeping for {retry} seconds...')
                time.sleep(retry)

                # Retry once
                res = session.get(url, stream=True)
                if res.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in res.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                print(f'Failed to download {filename} (Status: {res.status_code})')

        except Exception as e:
            print(f'Network error on {filename}: {e}')

        # Polite Concurrency Contract
        # Pausing ~2s safeguards against automated IP block triggers
        time.sleep(random.uniform(1.0, 3.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Model Generator'
    )

    parser.add_argument('-q', '--query')
    parser.add_argument('-n', '--count', type=int)

    parser.add_argument('-c', '--clean', action='store_true')

    parser.add_argument('-e', '--encoder', action='store_true')
    parser.add_argument('-o', '--output')
    parser.add_argument('-d', '--data')
    parser.add_argument('-s', '--image-size', type=int)

    parser.add_argument('-g', '--generator', action='store_true')
    parser.add_argument('-i', '--input')

    args = parser.parse_args()

    if args.query:
        n = args.count if args.count else 50
        urls = fetch_image_urls(query=args.query, count=n)
        download_images(urls, args.output)
    elif args.clean:
        from PIL import Image

        LIMIT = 50_000_000

        folder = Path(f'./data/{args.data}')

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            try:
                with Image.open(path) as img:
                    # Remove absurdly large images
                    if img.width * img.height > LIMIT:
                        print(f"Deleting (too large): {path}")
                        path.unlink()
                        continue

                    # Convert to a standard pixel format
                    if img.mode == "P":
                        # Palette image -> RGBA
                        img = img.convert("RGBA")
                    elif img.mode in ("RGBA", "LA"):
                        # Keep transparency
                        img = img.convert("RGBA")
                    else:
                        # RGB, L, CMYK, etc. -> RGB
                        img = img.convert("RGB")

                    # Overwrite the original file
                    img.save(path)

            except Exception as e:
                print(f"Deleting (corrupt): {path} ({e})")
                path.unlink()
    elif args.encoder:
        from autoencoder import save_model
        from torchvision import datasets
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(
            root=f'./data/{args.data}',
            transform=transform,
        )

        save_model(dataset=dataset, model_filename=f'./models/{args.output}')
    elif args.generator:
        from generator import save_generator

        save_generator(f'./models/{args.output}', f'./models/{args.input}')
