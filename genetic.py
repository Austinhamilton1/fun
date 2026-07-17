from typing import Any
from collections.abc import Callable, Iterator
import numpy as np
from PIL import Image

class Population:
    def __init__(
        self, 
        fitness_func: Callable[[np.ndarray], float],
        initial: list[list[Any]] = None, 
        n_size: int = 0, 
        spawner: Callable[[], list[Any]] = None,
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

    def __len__(self):
        return self.size
    
    def select(self, survival_rate: float) -> list[np.ndarray]:
        if survival_rate < 0.0 or survival_rate > 1.0:
            raise ValueError('Survival rate must be between 0.0 and 1.0')
        next_n_size = int(survival_rate * len(self.population))

        f = np.array([self.fitness(self.population[i]) for i in range(len(self.population))])
        return self.population[np.argsort(f)][-next_n_size:]
    
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
    
    def save(self):
        f = np.array([self.fitness(self.population[i]) for i in range(len(self.population))])
        order = np.argsort(f)
        self.most_fit = self.population[order[-1]]
        self.best_fitness = f[order[-1]]

generator = lambda: np.random.randint(0, 256, size=(400, 400), dtype=np.uint8)

def fitness(img):
    return (
        np.sum(img[:-1] == img[1:]) +
        np.sum(img[:, :-1] == img[:, 1:])
    )

def crossover(image1, image2):
    mask = np.random.choice([True, False], size=image1.shape, p=[0.75, 0.25])
    return np.where(mask, image1, image2)

def mutate(image):
    image = image.copy()
    mask = np.random.random(image.shape) < 0.02
    image[mask] = np.random.randint(0, 256, size=mask.sum(), dtype=np.uint8)

    return image

p = Population(fitness, n_size=20, spawner=generator, dtype=np.uint8)

for i in range(500):
    print('Iteration:', i+1)
    parents = p.select(0.4)
    p.breed(parents, crossover, mutate)
    p.save()

print(f'Best fitness: {p.best_fitness}')
image = Image.fromarray(p.most_fit, mode='L')
image.show()