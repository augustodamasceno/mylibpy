#!/usr/bin/env python3
"""
  Mylibpy Plot

  Copyright (c) 2025, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"


import numpy as np
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def scatter_with_continuos_3d(x_points,
                              y_points,
                              z_points,
                              fun,
                              limits: tuple=(-100, 100),
                              fun_name='Z function',
                              title='Scatter 3D with Continuous Surface',
                              filename='fun3d.html',
                              color_map='plasma'):

    space_search_grid = np.arange(limits[0], limits[1] + 5, 5)
    x, y = np.meshgrid(space_search_grid, space_search_grid)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, fun([x, y]), cmap=color_map, alpha=0.7)
    ax.scatter(x_points, y_points, z_points, c='red', marker='o', s=50, label='Points')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel(fun_name)
    ax.set_title(title)
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=5, label=fun_name)
    plt.show()

    surface = go.Surface(
        x=x,
        y=y,
        z=fun([x, y]),
        colorscale=color_map,
        opacity=0.7
    )
    scatter = go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            symbol='circle'
        ),
        name='Scatter Points'
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title=fun_name
        ),
        margin=dict(l=40, r=40, b=40, t=80)
    )
    fig.write_html(filename)