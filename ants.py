import numpy as np
from PIL import Image
from tqdm import tqdm

class Colony:
    def __init__(self, env, n_size: int = 1):
        self.env = env
        self.n_size = n_size
        self.ant_pos = np.array([[np.random.randint(s) for s in env.shape] for _ in range(n_size)])

    def update(self, pheremone_deposit: int = 100, decay: float = 0.999):
        offsets = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1), (1, 0), (1, 1)
        ])

        rows, cols = self.env.shape

        for i in range(self.n_size):
            r, c = self.ant_pos[i]

            # Deposit pheremone at current cell before moving on
            self.env[r, c] += pheremone_deposit

            # Wrap neighbor coordinates around the edge
            neighbor_coords = (self.ant_pos[i] + offsets) % [rows, cols]
            values = self.env[neighbor_coords[:, 0], neighbor_coords[:, 1]].astype(np.float64)

            total = values.sum()
            if total <= 0:
                probs = np.full(8, 1 / 8)
            else:
                probs = values / total

            choice = np.random.choice(8, p=probs)
            self.ant_pos[i] = neighbor_coords[choice]

        # After every ant has moved, the whole grid decays
        self.env *= decay
        np.clip(self.env, 0, 255, out=self.env)

data = np.random.randint(0, 256, size=(400, 400)).astype(np.float64)
colony = Colony(data, 500)

frames = []
n_frames = 1000
for _ in tqdm(range(n_frames)):
    colony.update()
    frame = Image.fromarray(data.astype(np.uint8), mode='L')
    frames.append(frame)

frames[0].save(
    'ants.gif',
    save_all=True,
    append_images=frames[1:],
    duration=50,
    loop=0,
)
