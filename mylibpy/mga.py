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
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


class GA:
    def __init__(self, **kwargs):
        self.verbose = False
        self.population_size = 10
        self.crossover_rate = 0.75
        self.mutation_rate = 0.01
        self.mutation_rate_start = 0.09
        self.mutation_rate_end = 0.01
        self.mutation_type = 'uniform'
        self.genetic_stall = 5
        self.max_generations = 20
        self.num_dimensions = 1
        self.range_low = -10
        self.range_high = 10
        self.range_interval = 20
        self.num_digits = 19
        self.bin_max_val = 4294967295
        self.selection_type = 'roulette'
        self.crossover_type = 'two'
        self.elit = 2
        self.pop = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.best_fitness_history = []
        self.run_start_time = None
        self.run_end_time = None
        self.fitness_func = None
        self.generation = 0
        self.fitness_func_counter = 0
        self.save_plots = False
        self.output_dir = "."
        self.save_only_last_plot = False
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def set_codification(self):
        if self.range_high < self.range_low:
            raise ValueError('range_high must be higher than range_low')

        self.range_interval = np.abs(self.range_high - self.range_low)
        self.bin_max_val =  2 ** self.num_digits - 1

    def init(self):
        self.set_codification()
        self.generation = -1
        self.fitness_func_counter = 0
        if self.mutation_type == 'non-uniform':
            self.mutation_rate = self.mutation_rate_start
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

        roulette_fitness = self.fitness + np.abs(min_val) if min_val < 0 else self.fitness
        fitness_sum = np.sum(roulette_fitness)
        if fitness_sum == 0:
            slices = np.ones(self.population_size) / self.population_size
        else:
            slices = roulette_fitness / fitness_sum
        slices_end = np.cumsum(slices)

        selected = []
        num_couples = int(self.population_size * self.crossover_rate) // 2
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
            ) and attempts < 30:
                father = GA.spin_wheel(slices_end)
                mother = GA.spin_wheel(slices_end)
                attempts += 1
            if attempts < 30:
                selected.append((father, mother))
                parents.add(father)
                parents.add(mother)
            else:
                break
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
            self.fitness_func_counter += 1

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
        if len(children) == 0:
            return children
        if self.mutation_type in ['uniform', 'non-uniform']:
            if self.mutation_type == 'non-uniform':
                self.mutation_rate -= (self.mutation_rate_start - self.mutation_rate_end)/(self.max_generations+1)
                if self.verbose:
                    print(f'Mutation Rate = {self.mutation_rate}')
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

    def save_generation_plot(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        decoded_pop = np.array([self.decode(ind) for ind in self.pop])
        
        if self.num_dimensions == 1:
            fig, ax = plt.subplots(figsize=(10, 12))
            fitness_values = np.array([self.fitness_func(val) for val in decoded_pop])
            ax.scatter(decoded_pop, fitness_values, c='red', s=50, alpha=0.6, label='Population')
            x_range = np.linspace(self.range_low, self.range_high, 500)
            y_range = np.array([self.fitness_func(x) for x in x_range])
            ax.plot(x_range, y_range, 'b-', linewidth=2, label='Fitness Function')
            ax.set_xlabel('X')
            ax.set_ylabel('Fitness')
            ax.set_title(f'Generation {self.generation} - Best Fitness: {self.best_fitness:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = f'{self.output_dir}generation_{self.generation:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
        elif self.num_dimensions == 2:
            # Create 3D plot for 2D problems
            fig = plt.figure(figsize=(14, 14))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=15, azim=100)
            
            # Plot the fitness function surface
            x_range = np.linspace(self.range_low, self.range_high, 50)
            y_range = np.linspace(self.range_low, self.range_high, 50)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.fitness_func(np.array([X[i, j], Y[i, j]]))
            
            ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6, edgecolor='none')
            
            # Plot the population points
            fitness_values = np.array([self.fitness_func(val) for val in decoded_pop])
            ax.scatter(decoded_pop[:, 0], decoded_pop[:, 1], fitness_values, 
                      c='red', s=100, alpha=0.8, label='Population', edgecolors='black', linewidths=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Fitness')
            ax.set_title(f'Generation {self.generation} - Best Fitness: {self.best_fitness:.4f}')
            ax.legend()
            filename = f'{self.output_dir}generation_{self.generation:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(10, 12))
            ax.scatter(decoded_pop[:, 0], decoded_pop[:, 1], c='red', s=50, alpha=0.6, label='Population')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'Generation {self.generation} - Best Fitness: {self.best_fitness:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = f'{self.output_dir}generation_{self.generation:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()


    def run(self, fitness_func):
        if inspect.isfunction(fitness_func):
            self.fitness_func = fitness_func
        else:
            raise ValueError("fitness_func must be a function.")

        self.best_solution = None
        self.best_fitness = -np.inf
        self.init()
        self.run_start_time = time.perf_counter()
        while not self.is_stop_criteria_reached():
            self.generation += 1
            if self.verbose:
                print(f'Generation {self.generation}')
                for chromosome in self.pop:
                    print(f'\t{chromosome}')

            self.population_fitness()
            self.best_fitness_history.append(self.best_fitness)

            if self.save_plots and not self.save_only_last_plot:
                self.save_generation_plot()

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
        self.run_end_time = time.perf_counter()
        elapsed_time = self.run_end_time - self.run_start_time
        if self.verbose:
            print(f'Fitness History:\n{[ float(x) for x in self.best_fitness_history]}\n'
                  + f'End at generation {self.generation-1}\n'
                  + f'End Population {len(self.pop)}\n'
                  + f"Elapsed time: {elapsed_time:.4f} seconds\n"
                  + f"Fitness Function Calls: {self.fitness_func_counter}"
                  )
        if self.save_only_last_plot:
            self.save_generation_plot()


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
            num_dimensions=1,
            genetic_stall=20,
            crossover_type='two',
            selection_type='roulette',
            mutation_type='non-uniform',
            mutation_rate=0.08,
            mutation_rate_start = 0.01,
            mutation_rate_end = 0.1,
            range_low=-10,
            range_high=10,
            num_digits=19,
            verbose=True,
            save_plots=False,
            output_dir="../outputs/ga-plots/")
    print(f'Running GA Example with config\n{ga}\n')

    ga.run(fitness_func=lambda x: -x*x + 3*x - 4)
    print(f'Best Solution: {ga.decode(ga.best_solution)}\nBest Fitness: {ga.best_fitness}')

