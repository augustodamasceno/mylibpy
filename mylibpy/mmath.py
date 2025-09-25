#!/usr/bin/env python3
"""
  Mylibpy Math

  Copyright (c) 2023, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"

import numpy as np

def exponential_decay(val_min, val_max, period, index):
    close_zero = 0.000001
    if val_min == 0:
        tau = period / np.log(val_max / close_zero)
    else:
        tau = period / np.log(val_max / val_min)

    exp = -index / tau
    value = val_max * np.exp(exp)

    if val_min == 0:
        value -= close_zero

    return value


if __name__ == "__main__":
    pass
