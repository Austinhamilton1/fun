import numpy as np
from PIL import Image
from tqdm import tqdm

class Bee:
    def __init__(self, env):
        self.env = env
        self.pollen = 0

    def update(self, pos):
        if self.env[pos] <= self.pollen:
            max_drop = 255 - self.env[pos]
            drop = min(max_drop, self.pollen)
            self.env[pos] += drop
            self.pollen -= drop
        else:
            max_pull = 255 - self.pollen
            pull = min(max_pull, self.env[pos])
            self.pollen += pull
            self.env[pos] -= pull

class Hive:
    def __init__(self, env, n_size: int = 1):
        self.env = env
        self.n_size = n_size
        self.bee_pos = np.array([[np.random.uniform(s) for s in env.shape] for _ in range(n_size)])
        self.bee_vel = np.array([[np.random.uniform(-6, 6) for _ in env.shape] for _ in range(n_size)])
        self.bees = [Bee(self.env) for _ in range(n_size)]

    def update(self):
        average_position = np.average(self.bee_pos, axis=0)
        average_velocity = np.average(self.bee_vel, axis=0)

        # Fly towards the flock's center of mass
        cohesion = (average_position - self.bee_pos) * 0.005

        # Steer toward the flock's average heading
        alignment = (average_velocity - self.bee_vel) * 0.05

        # Steer away from other bees to avoid crowding
        separation = np.zeros_like(self.bee_pos)
        for i in range(self.n_size):
            diffs =  self.bee_pos[i] - self.bee_pos
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            close = dists < 20
            if np.any(close):
                push = diffs[close] / (dists[close, None] + 1e-6)
                separation[i] = np.sum(push, axis=0) * 0.5

        # Small random jitter so the swarm doesn't settle into a static pattern
        jitter = np.random.uniform(-2.0, 2.0, size=self.bee_vel.shape)

        self.bee_vel += cohesion + alignment + separation + jitter
        self.bee_vel = np.clip(self.bee_vel, -6, 6)
        self.bee_pos += self.bee_vel

        for dim, size in enumerate(self.env.shape):
            self.bee_pos[:, dim] = self.bee_pos[:, dim] % size

        for i in range(self.n_size):
            self.bees[i].update(tuple(self.bee_pos[i].astype(np.int64)))

data = np.random.randint(0, 256, size=(400, 400))
hive = Hive(data, 200)

while True:
    try:
        hive.update()
    except KeyboardInterrupt:
        break

data = data.astype(np.uint8)
image = Image.fromarray(data, mode='L')
image.save('bees.png')
