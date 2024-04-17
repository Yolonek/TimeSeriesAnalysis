import mplcyberpunk
import numpy as np
from matplotlib.collections import PolyCollection


def enhance_plot(figure, axes, glow=False, alpha_gradient=0, lines=True):
    figure.set_facecolor('black')
    axes.set_facecolor('black')
    if glow:
        if lines:
            mplcyberpunk.make_lines_glow(ax=axes)
        else:
            mplcyberpunk.make_scatter_glow(ax=axes)
    if 1 > alpha_gradient > 0:
        mplcyberpunk.add_gradient_fill(ax=axes, alpha_gradientglow=alpha_gradient)


def set_polygon_color(axes, color):
    for item in axes.collections:
        if isinstance(item, PolyCollection):
            item.set_facecolor(color)


class AutoRegressiveModel:
    def __init__(self,
                 coefficients: list[float:],
                 constant: float = 0):
        self.coefficients = np.array(coefficients)
        self.order = len(coefficients)
        self.constant = constant

    def __str__(self):
        coeffs = [f'{theta}y_{r"{"}t-{i + 1}{r"}"}'
                  for i, theta in enumerate(self.coefficients)
                  if theta != 0]
        coeffs = ' + '.join(coeffs)
        return f'$y_t ={f" {self.constant} +" if self.constant != 0 else ""} {coeffs} + \epsilon_t$'

    def __call__(self, n: int, time_series: np.array = None, burn_in: int = 0):
        return self._fit(n, time_series=time_series, burn_in=burn_in)

    def _fit(self, n: int, time_series: np.array = None, burn_in: int = 0):
        pred_len = self.order + n + burn_in
        prediction = np.zeros(pred_len)
        noise = np.random.normal(loc=0, scale=1, size=pred_len)
        if time_series is None:
            prediction[:self.order] = noise[:self.order]
        else:
            prediction[:self.order] = time_series[::-1][:self.order]
        for t in range(self.order, pred_len):
            prediction[t] = (np.sum(prediction[t - self.order:t][::-1] * self.coefficients)
                             + self.constant + noise[t])
        return prediction[self.order + burn_in:]


class MovingAverageModel:
    def __init__(self,
                 coefficients: list[float:],
                 constant: float = 0):
        self.coefficients = np.array(coefficients)
        self.order = len(coefficients)
        self.constant = constant

    def __str__(self):
        coeffs = [f'{theta}\epsilon_{r"{"}t-{i + 1}{r"}"}'
                  for i, theta in enumerate(self.coefficients)
                  if theta != 0]
        coeffs = ' + '.join(coeffs)
        return f'$y_t ={f" {self.constant} +" if self.constant != 0 else ""} {coeffs} + \epsilon_t$'

    def __call__(self, n: int, time_series: np.array = None, burn_in: int = 0):
        return self._fit(n, time_series=time_series, burn_in=burn_in)

    def _fit(self, n: int, time_series: np.array = None, burn_in: int = 0):
        pred_len = self.order + n + burn_in
        prediction = np.zeros(pred_len)
        noise = np.random.normal(loc=0, scale=1, size=pred_len)
        if time_series is None:
            prediction[:self.order] = noise[:self.order]
        else:
            prediction[:self.order] = time_series[::-1][:self.order]
        for t in range(self.order, pred_len):
            prediction[t] = (np.sum(noise[t - self.order:t][::-1] * self.coefficients)
                             + self.constant + noise[t])
        return prediction[self.order + burn_in:], noise[self.order + burn_in:]
