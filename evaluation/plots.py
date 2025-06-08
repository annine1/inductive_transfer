from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def percentile_plot(y: np.ndarray,
                    y_hat: np.ndarray,
                    title: str = '') -> Tuple[mpl.figure.Figure, mpl.axes._subplots.Axes]:
    """Plot the time series of observed values with 3 specific prediction intervals (i.e.: 25 to 75, 10 to 90, 5 to 95).

    Parameters
    ----------
    y : np.ndarray
        Array of observed values.
    y_hat : np.ndarray
        Array of simulated values, where the last dimension contains the samples for each time step.
    title : str, optional
        Title of the plot.

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axis]
        The percentile plot.
    """
    fig, ax = plt.subplots()

    y_median = np.median(y_hat, axis=-1).flatten()
    y_25 = np.percentile(y_hat, 25, axis=-1).flatten()
    y_75 = np.percentile(y_hat, 75, axis=-1).flatten()
    y_10 = np.percentile(y_hat, 10, axis=-1).flatten()
    y_90 = np.percentile(y_hat, 90, axis=-1).flatten()
    y_05 = np.percentile(y_hat, 5, axis=-1).flatten()
    y_95 = np.percentile(y_hat, 95, axis=-1).flatten()

    x = np.arange(len(y_05))

    ax.fill_between(x, y_05, y_95, color='#35B779', label='05-95 PI')
    ax.fill_between(x, y_10, y_90, color='#31688E', label='10-90 PI')
    ax.fill_between(x, y_25, y_75, color="#440154", label='25-75 PI')
    ax.plot(y_median, '-', color='red', label="median")
    ax.plot(y.flatten(), '--', color='black', label="observed")
    ax.legend()
    ax.set_title(title)

    return fig, ax


def regression_plot(y: np.ndarray,
                    y_hat: np.ndarray,
                    title: str = '') -> Tuple[mpl.figure.Figure, mpl.axes._subplots.Axes]:
    """Plot the time series of observed and simulated values.

    Parameters
    ----------
    y : np.ndarray
        Array of observed values.
    y_hat : np.ndarray
        Array of simulated values.
    title : str, optional
        Title of the plot.

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axis]
        The regression plot.
    """

    fig, ax = plt.subplots()

    ax.plot(y.flatten(), label="observed", lw=1)
    ax.plot(y_hat.flatten(), label="simulated", alpha=.8, lw=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title(title)

    return fig, ax
