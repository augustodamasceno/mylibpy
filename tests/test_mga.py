#!/usr/bin/env python3
"""
  Mylibpy Genetic Algorithm Unit Tests

  Copyright (c) 2023, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"

import unittest
import numpy as np

from mylibpy.mga import GA


class TestGA(unittest.TestCase):
    def test_decode(self):
        ga = GA(range_low=-1,
                range_high=1,
                decimal_places=1)
        ga.set_codification()
        max_val = 1.0
        min_val = -1.0

        decoded = ga.decode([1, 1, 1, 1, 1])
        self.assertEqual(max_val, decoded)

        decoded = ga.decode([0, 0, 0, 0, 0])
        self.assertEqual(min_val, decoded)

        decoded_a = ga.decode([0, 1, 1, 1, 1])
        decoded_b = ga.decode([1, 0, 0, 0, 0])
        self.assertEqual(decoded_a, -1*decoded_b)

        min_inc = ga.range_interval / ga.bin_max_val
        self.assertAlmostEqual(min_inc, 2*decoded_b, 15)


    def test_accuracy(self):
        ga = GA(population_size=10,
                max_generations=20,
                crossover_rate=0.85,
                elit=2,
                genetic_stall=20,
                crossover_type='two',
                selection_type='roulette',
                mutation_type='uniform',
                mutation_rate=0.1,
                range_low=-10,
                range_high=10,
                decimal_places=12,
                verbose=False)

        best_fitness_history = []
        for i in range(100):
            ga.run(fitness_func=lambda x: -x * x + 3 * x - 4)
            best_fitness_history.append(ga.best_fitness)

        desired_value = -1.75
        accepted_error = 10**-4
        accepted_accuracy = 0.8
        best_fitness_history = np.array(best_fitness_history)
        error = np.abs(desired_value-best_fitness_history)
        acceptable = error <= accepted_error
        accuracy = np.sum(acceptable)/len(best_fitness_history)

        self.assertGreaterEqual(accuracy, accepted_accuracy)


if __name__ == '__main__':
    unittest.main()
