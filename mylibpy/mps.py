#!/usr/bin/env python3
"""
  Mylibpy Particle Swarm

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

try:
    from mplot import scatter_with_continuos_3d
except:
    from .mplot import scatter_with_continuos_3d


class PS:
    def __init__(self, **kwargs):
        self.verbose = False
        self.num_stall = 5
        self.num_particles = 20
        self.max_iterations = 20
        self.num_dimensions = 1
        self.range_low = -10
        self.range_high = 10
        self.inertia_max = 0.9
        self.inertia_min = 0.1
        self.delta_inertia = 0.8
        self.inertia_rate = 0.4
        self.cognitive_behaviour = 2
        self.social_behaviour = 2
        self.position = None
        self.velocity = None
        self.personal_best = None
        self.global_best = None
        self.personal_eval = None
        self.global_eval = None
        self.run_start_time = None
        self.run_end_time = None
        self.elapsed_time = np.nan
        self.func = None
        self.iteration = 0
        self.func_counter = 0
        self.save_plots = False
        self.output_dir = "."
        self.save_only_last_plot = False
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def inertia(self):
        return self.inertia_max - self.iteration  * self.inertia_rate

    def init(self):
        self.delta_inertia = self.inertia_max - self.inertia_min
        self.inertia_rate = self.delta_inertia / self.max_iterations
        self.iteration = -1
        self.func_counter = 0
        self.position = np.random.randint(self.range_low,
                                          self.range_high,
                                    (self.max_iterations, self.num_particles, self.num_dimensions))
        self.personal_best = np.random.randint(self.range_low,
                                               self.range_high,
                                          (self.max_iterations, self.num_particles, self.num_dimensions))
        self.personal_eval = np.zeros((self.max_iterations, self.num_particles))
        self.global_eval = np.zeros(self.max_iterations)
        self.global_best = np.zeros((self.max_iterations, self.num_dimensions))
        self.velocity =  np.zeros((self.max_iterations, self.num_particles))

    def update_velocity(self):
        current_inertia = self.inertia()
        r1 = np.random.random()
        r2 = np.random.random()
        cognitive = self.cognitive_behaviour * r1 * (self.personal_best[self.iteration] - self.position[self.iteration])
        social = self.social_behaviour * r2 * (self.global_best - self.position[self.iteration])
        self.velocity = current_inertia * cognitive + social

    def update_position(self):
        self.position[self.iteration+1] = self.position[self.iteration] + self.velocity
        self.position[self.iteration+1] = np.where(self.position[self.iteration+1]  > self.range_high, self.range_high, self.position[self.iteration+1] )
        self.position[self.iteration+1] = np.where(self.position[self.iteration+1]  < self.range_low, self.range_low, self.position[self.iteration+1] )

    def update_best(self):
        max_global_eval = -np.inf if self.iteration == 0 else self.global_eval[self.iteration-1]
        for particle in range(self.num_particles):
            particle_position = self.position[self.iteration, particle]
            particle_eval = self.func(particle_position)
            self.personal_eval[self.iteration, particle] = particle_eval
            if self.iteration == 0:
                self.personal_best[self.iteration, particle] = particle_position
            else:
                if particle_eval > self.personal_eval[self.iteration-1, particle]:
                    self.personal_best[self.iteration, particle] = particle_position
                else:
                    self.personal_best[self.iteration, particle] = self.personal_eval[self.iteration-1, particle]
            if particle_eval > max_global_eval:
                max_global_eval = particle_eval
        self.global_eval[self.iteration] = max_global_eval


    def save_iteration_plot(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.num_dimensions == 1:
            fig, ax = plt.subplots(figsize=(10, 12))
            fitness_values = np.array([self.func(val) for val in self.personal_eval])
            ax.scatter(self.personal_eval, fitness_values, c='red', s=50, alpha=0.6, label='Particles')
            x_range = np.linspace(self.range_low, self.range_high, 500)
            y_range = np.array([self.func(x) for x in x_range])
            ax.plot(x_range, y_range, 'b-', linewidth=2, label='Values')
            ax.set_xlabel('X')
            ax.set_ylabel('Values')
            ax.set_title(f'Iteration {self.iteration}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = f'{self.output_dir}iteration{self.iteration:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
        elif self.num_dimensions == 2:
            fig = plt.figure(figsize=(14, 14))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=110, azim=10)
            x_range = np.linspace(self.range_low, self.range_high, 50)
            y_range = np.linspace(self.range_low, self.range_high, 50)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.func(np.array([X[i, j], Y[i, j]]))
            
            ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6, edgecolor='none')

            fitness_values = np.array([self.func(val) for val in self.position[-1]])
            ax.scatter(self.personal_eval[:, 0], self.personal_eval[:, 1], fitness_values, 
                      c='red', s=100, alpha=0.8, label='Particles', edgecolors='black', linewidths=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Value')
            ax.set_title(f'Iteration {self.iteration}')
            ax.legend()
            filename = f'{self.output_dir}iteration_{self.iteration:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()

            scatter_with_continuos_3d(self.position[-1, :, 0],
                                      self.position[-1, :, 1],
                                      fitness_values,
                                      self.func,
                                      limits=(self.range_low, self.range_high),
                                      fun_name='W30 + W4',
                                      title=f'W30 + W4 with Iteration {self.iteration}',
                                      filename=f'{self.output_dir}iteration{self.iteration:04d}.html',
                                      color_map='plasma')
        else:
            fig, ax = plt.subplots(figsize=(10, 12))
            ax.scatter(self.personal_eval[:, 0], self.personal_eval[:, 1], c='red', s=50, alpha=0.6, label='Particles')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'Generation {self.iteration}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = f'{self.output_dir}iteration_{self.iteration:04d}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()

    def is_stop_criteria_reached(self):
        return self.iteration >= self.max_iterations - 1

    def run(self, func):
        if inspect.isfunction(func):
            self.func = func
        else:
            raise ValueError("func must be a function.")


        self.init()
        self.run_start_time = time.perf_counter()
        while not self.is_stop_criteria_reached():
            self.update_best()
            self.update_velocity()
            self.update_position()
            self.func_counter += self.num_particles
            if self.save_plots and not self.save_only_last_plot:
                self.save_iteration_plot()
            self.iteration += 1

        self.run_end_time = time.perf_counter()
        self.elapsed_time = self.run_end_time - self.run_start_time
        if self.save_only_last_plot:
            self.save_iteration_plot()


    def __str__(self):
        lines = []
        for k, v in self.__dict__.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
        

if __name__ == "__main__":
    ps = PS(num_stall = 5,
            num_particles = 20,
            max_iterations = 20,
            num_dimensions = 1,
            range_low = -10,
            range_high = 10,
            inertia_max = 0.9,
            inertia_min = 0.1,
            delta_inertia = 0.8,
            inertia_rate = 0.4,
            cognitive_behaviour = 2,
            social_behaviour = 2,
            save_plots = False,
            save_only_last_plot = False,
            output_dir="../outputs/ps-plots/")
    print(f'Running PS Example with config\n{ps}\n')

    ps.run(func=lambda x: -x*x + 3*x - 4)
    print(f'Best Solution: {ps.global_best}')

