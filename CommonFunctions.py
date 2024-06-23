import mplcyberpunk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def enhance_plot(figure, axes, glow=False, alpha_gradient=0, lines=True, dpi=100):
    figure.set_facecolor('black')
    figure.set_dpi(dpi)
    axes.set_facecolor('black')
    for font in [axes.title, axes.xaxis.label, axes.yaxis.label]:
        font.set_fontweight('bold')
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


def plot_time_series_analysis(model_list, time_series_list, theta_params_list, lags, size, alpha, title, file):
    colors = ['red', 'lime', None, 'blueviolet']
    bg_colors = ['maroon', 'green', None, 'indigo']
    with plt.style.context('cyberpunk'):
        figure, axes = plt.subplot_mosaic(
            [['time', 'time'], *[[f'acf-{theta}', f'pacf-{theta}'] for theta in theta_params_list]],
            layout='constrained', figsize=size,
            height_ratios=[2, *[1 for _ in range(len(time_series_list))]])
        for index, (theta_params, time_series, model) in (
                enumerate(zip(theta_params_list, time_series_list, model_list))):
            num_of_series = len(time_series_list)
            time_domain = range(1, len(time_series) + 1)
            axes['time'].scatter(time_domain, time_series, s=1,
                                 label=str(model), alpha=alpha,
                                 color=colors[index])
            axes['time'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), labelspacing=0.1,
                                ncol=len(model_list), borderpad=0.1, markerscale=6)
            plot_acf(time_series, ax=axes[f'acf-{theta_params}'], lags=lags, title=None,
                     color=colors[index], vlines_kwargs=dict(color=bg_colors[index]))
            set_polygon_color(axes[f'acf-{theta_params}'], colors[index])
            plot_pacf(time_series, ax=axes[f'pacf-{theta_params}'], lags=lags, title=None,
                      color=colors[index], vlines_kwargs=dict(color=bg_colors[index]))
            set_polygon_color(axes[f'pacf-{theta_params}'], colors[index])
            axes['time'].set(ylabel='Time Series')
            if index == num_of_series - 1:
                axes[f'acf-{theta_params}'].set(xlabel='lag')
                axes[f'pacf-{theta_params}'].set(xlabel='lag')
            if index == 0:
                axes[f'acf-{theta_params}'].set(title='Autocorrelation')
                axes[f'pacf-{theta_params}'].set(title='Partial Autocorrelation')
            enhance_plot(figure, axes[f'acf-{theta_params}'])
            enhance_plot(figure, axes[f'pacf-{theta_params}'])
        figure.suptitle(title)
        enhance_plot(figure, axes=axes['time'])
    figure.savefig(file)


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
