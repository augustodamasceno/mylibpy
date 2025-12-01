#!/usr/bin/env python3
"""
  Mylibpy Benchmark Functions for Optimization Tests

  Copyright (c) 2025, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"

import numpy as np

try:
    from mplot import scatter_with_continuos_3d
except:
    from .mplot import scatter_with_continuos_3d


space_search_grid = np.arange(-500, 505, 5)
x, y = np.meshgrid(space_search_grid, space_search_grid)
z = -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))
x = x / 250.0
y = y / 250.0
r = 100 * (y - x**2)**2 + (1 - x)**2
r1 = (y - x**2)**2 + (1 - x)**2
rd = 1 + r1
x1 = 25 * x
x2 = 25 * y
xs = np.arange(-10, 10 + 0.1, 0.1)
ys = np.arange(-10, 10 + 0.1, 0.1)
a = 500
b = 0.1
c = 0.5 * np.pi
F10 = -a * np.exp(-b * np.sqrt((x1**2 + x2**2) / 2)) - np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)
XS, YS = np.meshgrid(xs, ys, indexing='ij')
R_sq = XS**2 + YS**2
zsh = 0.5 - ((np.sin(np.sqrt(R_sq)))**2 - 0.5) / (1 + 0.1 * R_sq)**2
Fobj = F10 * zsh
w = r * z
w1 = r + z
w2 = z - r1
w3 = r - z
w4ant = np.sqrt(r**2 + z**2)
w4 = np.sqrt(r**2 + z**2) + Fobj
w5 = w - 0.5 * w1
w6 = w + w2
w7 = w1 + w4
w8 = w1 - w4
w9 = w2 - w4
w10 = w2 + w4
w11 = w3 - w4
w12 = r + w4 * np.cos(y)
w13 = np.sqrt(w1) + np.sqrt(w3) - w4ant * np.cos(x)
w14 = z * np.exp(np.sin(r1))
w15 = z * np.exp(np.cos(r1))
w16 = w14 + w4
w17 = -w14 + w4
w18 = -w15 + w4
w19 = np.exp(-r1) * z
w20 = x * z
w21 = (x + y) * z
w22 = (x - y) * z
w23 = z / rd
w24 = (x - y) * w23
w25 = (x + y) * w23
w26 = -w4 / rd
w27 = w4 + w23
w28 = w4 - w23
w29 = w14 + w23
w30 = w4 + w14 + w23
w31 = w21 + w22
w32 = w21 + w23
w33 = w22 + w25
w34 = w22 + w26
w35 = w23 + w27
w36 = w23 + w28
w37 = w23 + w29
w38 = w23 + w30
w39 = w25 + w30
w40 = w27 + w30
x = x * 250
y = y * 250


def w30_add_w4_min(input):
    result = w30_add_w4(input)
    return -result


def w30_add_w4(input):
    assert len(input) == 2, 'input must be bidimensional'

    x = input[0]
    y = input[1]
    z = -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))
    x_scaled = x / 250.0
    y_scaled = y / 250.0
    r = 100 * (y_scaled - x_scaled ** 2) ** 2 + (1 - x_scaled) ** 2
    r1 = (y_scaled - x_scaled ** 2) ** 2 + (1 - x_scaled) ** 2
    rd = 1 + r1
    x1 = 25 * x_scaled
    x2 = 25 * y_scaled
    a = 500
    b = 0.1
    c = 0.5 * np.pi
    F10 = -a * np.exp(-b * np.sqrt((x1 ** 2 + x2 ** 2) / 2)) - np.exp((np.cos(c * x1) + np.cos(c * x2)) / 2) + np.exp(1)
    R_sq = x1**2 + x2**2
    zsh = 0.5 - ((np.sin(np.sqrt(R_sq))) ** 2 - 0.5) / (1 + 0.1 * R_sq) ** 2
    Fobj = F10 * zsh
    w = r * z
    w1 = r + z
    w2 = z - r1
    w3 = r - z
    w4ant = np.sqrt(r ** 2 + z ** 2)
    w4 = np.sqrt(r ** 2 + z ** 2) + Fobj
    w5 = w - 0.5 * w1
    w6 = w + w2
    w7 = w1 + w4
    w8 = w1 - w4
    w9 = w2 - w4
    w10 = w2 + w4
    w11 = w3 - w4
    w12 = r + w4 * np.cos(y_scaled)
    w13 = np.sqrt(w1) + np.sqrt(w3) - w4ant * np.cos(x_scaled)
    w14 = z * np.exp(np.sin(r1))
    w15 = z * np.exp(np.cos(r1))
    w16 = w14 + w4
    w17 = -w14 + w4
    w18 = -w15 + w4
    w19 = np.exp(-r1) * z
    w20 = x_scaled * z
    w21 = (x_scaled + y_scaled) * z
    w22 = (x_scaled - y_scaled) * z
    w23 = z / rd
    w24 = (x_scaled - y_scaled) * w23
    w25 = (x_scaled + y_scaled) * w23
    w26 = -w4 / rd
    w27 = w4 + w23
    w28 = w4 - w23
    w29 = w14 + w23
    w30 = w4 + w14 + w23
    w31 = w21 + w22
    w32 = w21 + w23
    w33 = w22 + w25
    w34 = w22 + w26
    w35 = w23 + w27
    w36 = w23 + w28
    w37 = w23 + w29
    w38 = w23 + w30
    w39 = w25 + w30
    w40 = w27 + w30
    
    return w30 + w4


if __name__ == "__main__":
    print('Plotting W30 + W4')

    np.random.seed(42) 
    n_points = 50
    random_x = np.random.uniform(-500, 500, n_points)
    random_y = np.random.uniform(-500, 500, n_points)
    random_z = np.array([w30_add_w4([rx, ry]) for rx, ry in zip(random_x, random_y)])
    scatter_with_continuos_3d(random_x,
                              random_y,
                              random_z,
                              w30_add_w4,
                              limits=(-500, 500),
                              fun_name='W30 + W4',
                              title='Scatter 3D with Continuous Surface',
                              filename='../outputs/benchmark-w30w4.html',
                              color_map='plasma')
    