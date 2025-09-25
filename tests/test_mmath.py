#!/usr/bin/env python3
"""
  Mylibpy Math Unit Tests

  Copyright (c) 2023, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"

import unittest

from mylibpy import mmath


class MyTestCase(unittest.TestCase):
    def test_exponential_decay(self):
        test_cases = [
            (0.1, 1.0, 100, 0, 1.0),
            (0, 1.0, 100, 0, 1.0),
            (0.1, 1.0, 100, 99, 0.1),
            (0, 1.0, 100, 99, 0.0),
            (0.1, 3.0, 100, 99, 0.1),
            (0.1, 3.0, 100, 0, 3.0),
            (0.1, 1.0, 3, 2, 0.2154435),
        ]

        for val_min, val_max, period, index, expected_value in test_cases:
            value = mmath.exponential_decay(val_min, val_max, period, index)
            self.assertAlmostEqual(value, expected_value, places=2,
                                   msg=(f"DecayExp({index}, {val_min},"
                                       +f" {val_max}, {period}) "
                                        f"should be {expected_value}"))

if __name__ == '__main__':
    unittest.main()
