# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
  Mylibpy Datetime

  Copyright (c) 2025, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "1.0"
__copyright__ = "Copyright (c) 2025, Augusto Damasceno."
__license__ = "BSD-2-Clause"


import inspect
import warnings
import random

import numpy as np

class GA:
    def __init__(self, **kwargs):
        self.verbose = False
        self.population_size = 32
        self.crossover_rate = 0.75
        self.mutation_rate = 0.01
        self.genetic_stall = 5
        self.max_generations = 20
        self.num_dimensions = 1
        self.bits_per_decimal_place = 3.3
        self.range_low = -10
        self.range_high = 10
        self.range_interval = 20
        self.decimal_places = 6
        self.num_digits = 32
        self.bin_max_val = 4294967295
        self.selection_type = 'roulette'
        self.crossover_type = 'two'
        self.pop = None
        self.fitness = None
        self.next_pop = None
        self.best_solution = None
        self.best_fitness = -np.inf
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

    def init_population(self):
        self.set_codification()

        if self.num_dimensions == 1:
            self.pop = np.random.randint(0, 2, (self.population_size,
                                                self.num_digits))
        else:
            self.pop = np.random.randint(0, 2, (self.population_size,
                                                self.num_dimensions,
                                                self.num_digits))
        self.fitness = np.zeros(self.population_size)
        self.generation = 0

    def decode(self, chromosome):
        if self.num_dimensions == 1:
            chromosome_str = "".join(map(str, chromosome))
            dec = int(chromosome_str, 2)
            value = self.range_low + self.range_interval * dec / self.bin_max_val
        else:
            value = np.zeros(self.num_dimensions, self.num_digits)
            for dim in range(self.num_dimensions):
                chromosome_str = "".join(map(str, chromosome[dim]))
                dec = int(chromosome_str, 2)
                value[dim] = self.range_low + self.range_interval * dec / self.bin_max_val

        return value

    def roulette(self):
        min_val = np.min(self.fitness)
        max_val = np.max(self.fitness)
        max_val_idx = np.argmax(self.fitness)
        self.best_fitness = max_val
        self.best_solution = self.pop[max_val_idx]
        if min_val < 0:
            self.fitness += np.abs(min_val)+1
        fitness_sum = np.sum(self.fitness)
        slices = self.fitness / fitness_sum
        slices_end = np.cumsum(slices)

        selected = []
        for turn in range(int(self.population_size/2)):
            rnd = np.random.random()
            if rnd < self.crossover_rate:
                father = GA.spin_wheel(slices_end)
                mother = GA.spin_wheel(slices_end)
                while father == mother:
                    father = GA.spin_wheel(slices_end)
                    mother = GA.spin_wheel(slices_end)
                selected.append((father, mother))
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

    def alive(self, selected):
        unique = list(set(np.array(selected).flatten()))
        num_keep = self.population_size - len(selected) * 2
        if num_keep < 0:
            return []
        keep = [index for index in range(self.pop.shape[0]) if index not in unique]
        random.shuffle(keep)
        return keep[0:num_keep]

    def crossover(self, selected):
        if self.crossover_type == 'two':
            children = []
            slice = int(np.floor(self.num_digits/3))
            for couple in selected:
                father = self.pop[couple[0]]
                mother = self.pop[couple[1]]
                child_a = father
                child_b = mother
                child_a[slice:2*slice] = mother[slice:2*slice]
                child_b[slice:2 * slice] = father[slice:2 * slice]
                children.append(child_a)
                children.append(child_b)
            return np.array(children)
        else:
            warnings.warn(
                (f"Unavailable crossover {self.crossover_type},"
                 + " using default: two."),
                UserWarning
            )
            self.crossover_type = 'two'
            self.crossover(selected)

    @staticmethod
    def exponential_decay(val_min, val_max, period, index):
        if val_min <= 0:
            val_min = 0.1
        tau =  period / np.log(val_max/val_min)
        exp = -index / tau
        value = val_max * np.exp(exp)
        return value

    def mutation(self, children):
        for idx in range(children.shape[0]):
            rnd = np.random.random()
            if rnd < self.mutation_rate:
                rnd_index = np.random.randint(int(1E9))
                exponential = GA.exponential_decay(0.1, self.num_digits-1, 1E9, rnd_index)
                mutate_gene = int(np.round(exponential))
                children[idx][mutate_gene] = 1-children[idx][mutate_gene]
        return children

    def is_genetic_stall(self):
        return False

    def is_stop_criteria_reached(self):
        return (self.generation == self.max_generations
                or self.is_genetic_stall())

    def run(self, fitness_func):
        # Define Fitness Function
        if inspect.isfunction(fitness_func):
            self.fitness_func = fitness_func
        else:
            raise ValueError("fitness_func must be a function.")

        # Create Initial Population
        self.init_population()

        # Main Loop
        while not self.is_stop_criteria_reached():
            if self.verbose:
                print(f'Generation {self.generation}')

            self.population_fitness()
            if self.verbose:
                print(f'\tWorst Fitness {np.min(self.fitness)}\n'
                      + f'\tBest Fitness {np.max(self.fitness)}')

            selected = self.selection()
            if self.verbose:
                print(f'\tSelected {selected}')

            children = self.crossover(selected)
            children = self.mutation(children)

            if len(children) > 0:
                old_gen_keep_index = self.alive(selected)
                old_gen_keep = self.pop[old_gen_keep_index]
                if len(old_gen_keep) > 0:
                    new_pop = np.concat([children, old_gen_keep])
                else:
                    new_pop = children
                self.pop = new_pop
            self.generation += 1

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
            crossover_rate=0.75,
            crossover_type='two',
            selection_type='roulette',
            mutation_rate=0.01,
            range_low=-10,
            range_high=10,
            verbose=True)
    print(f'Running GA Example with config\n{ga}')
    ga.run(fitness_func=lambda x: -x*x + 3*x - 4)
    print(f'Best Solution: {ga.decode(ga.best_solution)}\nBest Fitness: {ga.best_fitness}')
