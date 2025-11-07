#!/usr/bin/env python3
"""
  Mylibpy Genetic Algorithm

  Copyright (c) 2025, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2025, Augusto Damasceno."
__license__ = "BSD-2-Clause"


import time
import inspect
import warnings
import random

import numpy as np

class GA:
    def __init__(self, **kwargs):
        self.verbose = False
        self.population_size = 10
        self.crossover_rate = 0.75
        self.mutation_rate = 0.01
        self.mutation_type = 'uniform'
        self.genetic_stall = 5
        self.max_generations = 20
        self.num_dimensions = 1
        self.bits_per_decimal_place = 3.3
        self.range_low = -10
        self.range_high = 10
        self.range_interval = 20
        self.decimal_places = 4
        self.num_digits = 32
        self.bin_max_val = 4294967295
        self.selection_type = 'roulette'
        self.crossover_type = 'two'
        self.elit = 2
        self.pop = None
        self.fitness = None
        self.next_pop = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.best_fitness_history = []
        self.run_start_time = None
        self.run_end_time = None
        self.fitness_func = None
        self.generation = 0
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def set_codification(self):
        if self.range_high < self.range_low:
            raise ValueError('range_high must be higher than range_low')

        self.range_interval = np.abs(self.range_high - self.range_low)
        integer_places = np.log2(self.range_interval)
        self.num_digits = int(
            np.ceil(
                integer_places
                + self.decimal_places
                * self.bits_per_decimal_place
            )
        )
        self.bin_max_val =  2 ** self.num_digits - 1

    def init(self):
        self.set_codification()
        self.generation = -1
        if self.genetic_stall < 2:
            warnings.warn(
                (f"Invalid genetic stall {self.genetic_stall},"
                 + " using default: 5."),
                UserWarning
            )
            self.genetic_stall = 5
        if self.elit < 1:
            warnings.warn(
                (f"Invalid elit {self.elit},"
                 + " using default: 2."),
                UserWarning
            )
            self.elit = 2
        if self.num_dimensions == 1:
            self.pop = np.random.randint(0, 2, (self.population_size,
                                                               self.num_digits))
        else:
            self.pop = np.random.randint(0, 2, (self.population_size,
                                                               self.num_dimensions,
                                                               self.num_digits))
        self.fitness = np.zeros(self.population_size)
        self.best_fitness_history = []

    def decode(self, chromosome):
        if self.num_dimensions == 1:
            chromosome_str = "".join(map(str, chromosome))
            dec = int(chromosome_str, 2)
            value = self.range_low + self.range_interval * dec / self.bin_max_val
        else:
            value = np.zeros(self.num_dimensions)
            for dim in range(self.num_dimensions):
                chromosome_str = "".join(map(str, chromosome[dim]))
                dec = int(chromosome_str, 2)
                value[dim] = self.range_low + self.range_interval * dec / self.bin_max_val

        return value

    def roulette(self):
        min_val = np.min(self.fitness)
        max_val = np.max(self.fitness)
        max_val_idx = np.argmax(self.fitness)
        if max_val > self.best_fitness:
            self.best_fitness = max_val
            self.best_solution = self.pop[max_val_idx].copy()
        if min_val < 0:
            self.fitness += np.abs(min_val)
        fitness_sum = np.sum(self.fitness)
        if fitness_sum == 0:
            slices = np.ones(self.population_size) / self.population_size
        else:
            slices = self.fitness / fitness_sum
        slices_end = np.cumsum(slices)

        selected = []
        num_couples = int(self.population_size / 2.0 * self.crossover_rate)
        parents = set()
        while len(selected) < num_couples:
            father = GA.spin_wheel(slices_end)
            mother = GA.spin_wheel(slices_end)
            attempts = 0
            while (father == mother
                   or (father, mother) in selected
                   or (mother, father) in selected
                   or father in parents
                   or mother in parents
            ) and attempts < 10:
                father = GA.spin_wheel(slices_end)
                mother = GA.spin_wheel(slices_end)
                attempts += 1
            if attempts <= 10:
                selected.append((father, mother))
                parents.add(father)
                parents.add(mother)
        return selected

    @staticmethod
    def spin_wheel(slices_end):
        rnd = np.random.random()
        compare = np.where(slices_end > rnd)
        if len(compare[0]) == 0:
            return 0
        first = compare[0][0]
        return first

    def selection(self) -> list:
        if self.selection_type == 'roulette':
            return self.roulette()
        else:
            warnings.warn(
                (f"Unavailable selection {self.selection_type},"
                + " using default: roulette."),
                UserWarning
            )
            self.selection_type = 'roulette'
            return self.selection()

    def population_fitness(self):
        for index in range(self.population_size):
            decoded_value = self.decode(self.pop[index])
            self.fitness[index] = self.fitness_func(decoded_value)

    def  elit_index(self):
        idx = np.argpartition(self.fitness, -self.elit)[-self.elit:]
        return idx

    def alive(self, selected):
        to_kill = np.array(list(set(np.array(selected).flatten())))
        best = self.elit_index()
        to_kill = to_kill[~np.isin(to_kill, best)]
        num_keep = self.population_size - len(selected) * 2 - self.elit
        if num_keep < 0:
            return []
        keep = [index for index in range(self.pop.shape[0]) if index not in to_kill]
        random.shuffle(keep)
        return keep[0:num_keep]

    def crossover(self, selected):
        if self.crossover_type == 'one':
            children = []
            for couple in selected:
                cut = np.random.randint(1, self.num_digits)
                father = self.pop[couple[0]].copy()
                mother = self.pop[couple[1]].copy()
                child_a = father.copy()
                child_b = mother.copy()
                child_a[cut:] = mother[cut:]
                child_b[cut:] = father[cut:]
                children.append(child_a)
                children.append(child_b)
                if self.verbose:
                    print(f'Parents:\n\t{father}\n\t{mother}\n'
                          + f'Children:\n\t{child_a}\n\t{child_b}')
            return np.array(children)
        elif self.crossover_type == 'two':
            children = []
            for couple in selected:
                cuts = np.sort(np.random.randint(1, self.num_digits, 2))
                cut1 = cuts[0]
                cut2 = cuts[1]
                fallback_one_point = cut2 == cut1

                if self.verbose:
                    print(f'Cuts at: {cut1} and {cut2}')

                if fallback_one_point:
                    father = self.pop[couple[0]].copy()
                    mother = self.pop[couple[1]].copy()
                    child_a = father.copy()
                    child_b = mother.copy()
                    child_a[cut1:] = mother[cut1:]
                    child_b[cut1:] = father[cut1:]
                    children.append(child_a)
                    children.append(child_b)
                    if self.verbose:
                        print(f'Parents:\n\t{father}\n\t{mother}\n'
                              + f'Children:\n\t{child_a}\n\t{child_b}')
                else:
                    father = self.pop[couple[0]].copy()
                    mother = self.pop[couple[1]].copy()
                    child_a = father.copy()
                    child_b = mother.copy()
                    child_a[cut1:cut2] = mother[cut1:cut2]
                    child_b[cut1:cut2] = father[cut1:cut2]
                    children.append(child_a)
                    children.append(child_b)
                    if self.verbose:
                        print(f'Parents:\n\t{father}\n\t{mother}\n'
                              + f'Children:\n\t{child_a}\n\t{child_b}')
            return np.array(children)
        else:
            warnings.warn(
                (f"Unavailable crossover {self.crossover_type},"
                 + " using default: two."),
                UserWarning
            )
            self.crossover_type = 'two'
            return self.crossover(selected)

    def mutation_mask(self, size):
        if self.num_dimensions == 1:
            rnd = np.random.random((size, self.num_digits))
        else:
            rnd = np.random.random((size, self.num_dimensions, self.num_digits))
        mask = rnd < self.mutation_rate
        return mask

    def mutation(self, children):
        if self.mutation_type == 'uniform':
            mask = self.mutation_mask(children.shape[0])
            locations = np.where(mask)
            children_copy = children.copy()
            children[mask] = 1 - children_copy[mask]

            if self.verbose:
                if len(locations[0]) == 0:
                    print('No Mutations Happens')
                else:
                    print(f'Mutations')
                    for chromosome_index in range(len(locations[0])):
                        print(f'Children {locations[0][chromosome_index]} '
                              + f'(genes {locations[1][chromosome_index]})\n'
                              + f'{children_copy[locations[0][chromosome_index]]}\n'
                              + '↓\n'
                              + f'{children[locations[0][chromosome_index]]}')
            return children
        else:
            warnings.warn(
                (f"Unavailable mutation_type {self.mutation_type},"
                 + " using default: uniform."),
                UserWarning
            )
            self.mutation_type = 'uniform'
            return self.mutation(children)

    def is_genetic_stall(self):
        diff = np.array(self.best_fitness_history[1:]) - np.array(self.best_fitness_history[0:-1])
        if len(diff) >= self.genetic_stall-1:
            stall_slice = diff[-self.genetic_stall:]
            count_zeros = np.sum(stall_slice == 0)
            if count_zeros == self.genetic_stall-1:
                return True
        return False

    def is_stop_criteria_reached(self):
        return (self.generation == self.max_generations
                or self.is_genetic_stall())

    def run(self, fitness_func):
        if inspect.isfunction(fitness_func):
            self.fitness_func = fitness_func
        else:
            raise ValueError("fitness_func must be a function.")

        self.best_solution = None
        self.best_fitness = -np.inf
        self.init()

        while not self.is_stop_criteria_reached():
            self.generation += 1
            if self.verbose:
                print(f'Generation {self.generation}')
                for chromosome in self.pop:
                    print(f'\t{chromosome}')

            self.population_fitness()
            self.best_fitness_history.append(self.best_fitness)
            if self.verbose:
                print('Fitness')
                for fit in self.fitness:
                    print(f'\t{fit}')
                print(f'\tWorst Fitness {np.min(self.fitness)}\n'
                      + f'\tBest Fitness {np.max(self.fitness)}')

            selected = self.selection()
            if self.verbose:
                print(f'\tSelected: ')
                for couple in selected:
                    print(f'\t\tParents {couple[0]} and {couple[1]}')

            children = self.crossover(selected)
            children = self.mutation(children)

            if len(children) > 0:
                best = self.elit_index()
                new_elit = self.pop[best].copy()
                old_gen_keep_index = self.alive(selected)
                old_gen_keep = self.pop[old_gen_keep_index]
                if len(old_gen_keep) > 0:
                    new_pop = np.concatenate([children, new_elit, old_gen_keep])
                else:
                    new_pop = np.concatenate([children, new_elit])
                self.pop = new_pop

        if self.verbose:
            print(f'Fitness History:\n{[ float(x) for x in self.best_fitness_history]}\n'
                  + f'End at generation {self.generation-1}\n'
                  + f'End Population {len(self.pop)}')

    def __str__(self):
        lines = []
        for k, v in self.__dict__.items():
            if k == 'pop':
                if v is not None:
                    lines.append(f"pop: shape {v.shape}")
                else:
                    lines.append("pop: None")
            elif k not in ['fitness'] and v is not None:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
        

if __name__ == "__main__":
    ga = GA(population_size=10,
                max_generations=20,
                crossover_rate=0.85,
                elit=2,
                genetic_stall=5,
                crossover_type='two',
                selection_type='roulette',
                mutation_type='uniform',
                mutation_rate=0.1,
                range_low=-10,
                range_high=10,
                decimal_places=6,
                verbose=True)
    print(f'Running GA Example with config\n{ga}\n')

    start_time = time.perf_counter()
    ga.run(fitness_func=lambda x: -x*x + 3*x - 4)
    end_time = time.perf_counter()

    print(f'Best Solution: {ga.decode(ga.best_solution)}\nBest Fitness: {ga.best_fitness}')
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
