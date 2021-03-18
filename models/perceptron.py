import numpy as np
import time
import random
from typing import Tuple, List
import matplotlib.pyplot as plt


class Perceptron():
    _weights: np.ndarray = np.array([])
    _b = 0.5
    _fig = plt.figure()
    _labels: np.ndarray
    _points: np.ndarray

    def __init__(
        self,
        data: np.ndarray
    ):
        self.data = data
        num_point_columns = len(self.data[0]) - 1
        self._labels = self.data[:, num_point_columns]
        self._points = np.delete(self.data, num_point_columns, 1)

    def _set_default_coefficients(self):
        np.random.seed(42)
        x = self.data[:, 1]
        x_max = max(x)
        for idx, _ in enumerate(self.data[0]):
            rand_weight = random.uniform(-1, 1)
            if idx == len(self.data[0]) - 1:
                self._b = np.random.rand(1)[0] + x_max
            else:
                self._weights = np.append(self._weights, rand_weight)

    def _get_data(self) -> np.ndarray:
        return self.data

    def _convert_linear_comb_to_line(self) -> Tuple[float, float]:
        W = self._weights
        b = self._b
        return (-W[0]/W[1], -b/W[1])

    def _get_line(self):
        return self._convert_linear_comb_to_line()

    @staticmethod
    def _step_function(t: np.ndarray) -> int:
        if t >= 0:
            return 1
        return 0

    def _adjust(
        self,
        learning_rate: float,
        epoch: int
    ) -> int:
        print("Epoch: ", str(epoch))
        valid_points = 0
        invalid_points = 0
        for idx in range(len(self._points)):
            point = self._points[idx]
            y_hat = self._step_function(
                np.matmul(point, self._weights) + self._b
            )
            if self._labels[idx] - y_hat == 1:
                for dim, _ in enumerate(self.data[0]):
                    if idx < len(self.data[0]) - 1:
                        invalid_points += 1
                        self._weights[dim] += self._points[idx][dim] * \
                            learning_rate
                self._b += learning_rate
            if self._labels[idx] - y_hat == -1:
                for dim, _ in enumerate(self.data[0]):
                    if idx < len(self.data[0]) - 1:
                        invalid_points += 1
                        self._weights[dim] -= self._points[idx][dim] * \
                            learning_rate
                self._b -= learning_rate
            else:
                valid_points += 1
        print("Valid Points: " + str(valid_points))
        print("Invalid Points: " + str(invalid_points))
        print(str(valid_points) + "/" + str(len(self._points)))
        print("--------------------------------------")
        return valid_points

    def plot_scatter(self):
        x = self.data[:, 0]
        x2 = self.data[:, 1]
        label = self.data[:, len(self.data[0]) - 1]
        if (len(self.data[0]) > 3):
            x3 = self.data[:, 2]
            plt.scatter(x, x2, x3, 100, label)
        else:
            plt.scatter(x, x2, 100, label)

    def plot_line(self):
        x = self.data[:, 0]
        coeffients = self._get_line()
        y = coeffients[0]*x+coeffients[1]
        plt.plot(x, y, '-r', label='split_line')

    def train(
        self,
        learning_rate=0.2,
        num_epochs=25,
        threshold=1
    ) -> bool:
        self._set_default_coefficients()
        print('Weights')
        print(self._weights)
        self.plot_scatter()
        for idx in range(num_epochs):
            valid_points = self._adjust(learning_rate, idx)
            if (valid_points / len(self._points)) >= threshold:
                print("DONE. All the points are split")
                break

        self.plot_line()
