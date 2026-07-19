from typing import Any
from collections.abc import Callable
import numpy as np
from PIL import Image
import zlib

class Population:
    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        initial: list[list[Any]] | None = None,
        n_size: int = 0,
        spawner: Callable[[], list[Any]] | None = None,
        dtype=np.float64,
    ):
        self.fitness = fitness_func
        if initial is not None:
            self.population = np.array([np.array(i, dtype=dtype) for i in initial])
        elif n_size > 0 and spawner is not None:
            self.population = np.array([np.array(spawner(), dtype=dtype) for _ in range(n_size)])
        else:
            raise ValueError('Must have either initial population or spawner')

        self.size = len(self.population)
        self.most_fit = None
        self.best_fitness = float('-inf')

    def __len__(self):
        return self.size

    def select(self, survival_rate: float) -> list[np.ndarray]:
        if survival_rate < 0.0 or survival_rate > 1.0:
            raise ValueError('Survival rate must be between 0.0 and 1.0')
        next_n_size = int(survival_rate * len(self.population))

        f = np.array([self.fitness(self.population[i]) for i in range(len(self.population))])
        order = np.argsort(f)
        best_idx = order[-1]
        if f[best_idx] > self.best_fitness:
            self.best_fitness = f[best_idx]
            self.most_fit = self.population[best_idx]
        return [ind for ind in self.population[np.argsort(f)][-next_n_size:]]

    def breed(
        self,
        parents: list[np.ndarray],
        crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mutate: Callable[[np.ndarray], np.ndarray],
    ):
        new_population = [parent for parent in parents]
        while len(new_population) < self.size:
            a, b = np.random.choice(len(parents), size=2, replace=False)
            offspring = crossover(parents[a], parents[b])
            offspring = mutate(offspring)
            new_population.append(offspring)

        self.population = np.array(new_population)

    def avg_fitness(self) -> float:
        f = np.array([self.fitness(self.population[i]) for i in range(len(self.population))])
        return np.average(f)

def generator():
    return np.random.randint(0, 256, size=(400, 400))

def fitness(img):
    img = img.astype(np.uint8)

    # 1. Shannon entropy of pixel value distribution (0 = flat, 1 = max variety)
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) / 8.0 # normalize by log2(256)

    # 2. Edge density: how much local structure/contrast exists
    dx = np.abs(np.diff(img.astype(np.int16), axis=1))
    dy = np.abs(np.diff(img.astype(np.int16), axis=0))
    edge_density = (dx.mean() + dy.mean()) / 2 / 255.0

    # 3. Symmetry: rewardd left-right mirror similarity
    mirrored = img[:, ::-1]
    symmetry = 1 - (np.abs(img.astype(np.int16) - mirrored).mean() / 255.0)

    # 4. Compressibility: structured/repetitive images compress well, pure noise does not
    compressed_size = len(zlib.compress(img.tobytes(), level=6))
    compressibility = 1 - (compressed_size / img.nbytes)

    return (
        0.35 * entropy +
        0.25 * edge_density +
        0.15 * symmetry +
        0.25 * compressibility
    )

def crossover(image1, image2):
    row = np.random.randint(1, image1.shape[0])
    return np.vstack([image1[:row], image2[row:]])

def mutate(image):
    image = image.copy()
    mask = np.random.random(image.shape) < 0.0005
    delta = np.random.randint(-5, 6, size=mask.sum())
    image = image.astype(np.int16)
    image[mask] += delta
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)

p = Population(fitness, n_size=100, spawner=generator, dtype=np.uint8)

i = 1
while True:
    try:
        parents = p.select(0.3)
        p.breed(parents, crossover, mutate)
        print(f'Current best fitness: {p.best_fitness}')
        i += 1
    except KeyboardInterrupt:
        break

print('Best fitness:', p.best_fitness)
img = Image.fromarray(p.most_fit, mode='L')
img.save('genetic.png')
