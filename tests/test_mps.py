#!/usr/bin/env python3
"""
  Mylibpy Particle Swarm Unit Tests

  Copyright (c) 2025, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"

import unittest
import numpy as np

from mylibpy.mps import PS
from mylibpy.mbenchmark import w30_add_w4_min, w30_add_w4


class TestPS(unittest.TestCase):

    def test_accuracy_benchmark(self):
        ps = PS(num_stall=5,
                num_particles=20,
                max_iterations=20,
                num_dimensions=2,
                range_low=-500,
                range_high=500,
                inertia_max=0.9,
                inertia_min=0.1,
                delta_inertia=0.8,
                inertia_rate=0.4,
                cognitive_behaviour=2,
                social_behaviour=2,
                save_plots=False,
                save_only_last_plot=True,
                output_dir="outputs/ps-plots/benchmark/")

        best_eval_history = []
        for i in range(1):
            ps.run(func=w30_add_w4_min)
            best_eval_history.append(ps.global_eval[-1])

        print(best_eval_history)
        desired_value = -1000
        accepted_error = 1
        accepted_accuracy = 0.75
        history = np.array(best_eval_history)
        error = np.abs(desired_value-history)
        acceptable = error <= accepted_error
        accuracy = np.sum(acceptable)/len(history)

        self.assertGreaterEqual(accuracy, accepted_accuracy)


if __name__ == '__main__':
    unittest.main()
