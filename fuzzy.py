import numpy as np
import matplotlib.pyplot as plt

class Fuzzy:
    def __init__(self):
        pass

    @staticmethod
    def triangular_membership_function(x, a, b, c):
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        return (
                np.clip((x - a) / (b - a + epsilon), 0, 1) * (a < x) * (x <= b) +
                np.clip((c - x) / (c - b + epsilon), 0, 1) * (b < x) * (x <= c)
        )
    @staticmethod
    def gaussian_membership_function(x, mean, sigma):
        exponent = -0.5 * ((x - mean) / (sigma + 1e-10)) ** 2  # Add a small epsilon to avoid division by zero
        return np.exp(exponent)

    @staticmethod
    def trapezoidal_membership_function(x_values, a, b, c, d):
        assert a <= b and b <= c and c <= d, 'abcd requires the four elements a <= b <= c <= d.'

        y = np.ones(len(x_values))

        idx = np.nonzero(x_values <= b)[0]
        y[idx] = np.maximum(0, np.minimum((x_values[idx] - a) / (b - a), 1))

        idx = np.nonzero(x_values >= c)[0]
        y[idx] = np.maximum(0, np.minimum((d - x_values[idx]) / (d - c), 1))

        idx = np.nonzero(x_values < a)[0]
        y[idx] = np.zeros(len(idx))

        idx = np.nonzero(x_values > d)[0]
        y[idx] = np.zeros(len(idx))

        return y

    @staticmethod
    def defuzz_centroid(x_values, membership_values):
        numerator = np.sum(x_values * membership_values)
        denominator = np.sum(membership_values)

        if denominator == 0:
            return np.nan  # Handle division by zero

        return numerator / denominator

    def run(self, function_name, x_values, params):
        if function_name == 'triangular':
            membership_values = self.triangular_membership_function(x_values, *params)
            title = 'Triangular Membership Function'
        elif function_name == 'gaussian':
            membership_values = self.gaussian_membership_function(x_values, *params)
            title = 'Gaussian Membership Function'
        elif function_name == 'trapezoidal':
            membership_values = self.trapezoidal_membership_function(x_values, *params)
            title = 'Trapezoidal Membership Function'
        else:
            raise ValueError("Invalid function_name. Use 'triangular', 'gaussian', or 'trapezoidal'.")

        # Plot the membership function
        plt.plot(x_values, membership_values, label=function_name.capitalize() + ' MF')
        plt.title(title)
        plt.xlabel('Input')
        plt.ylabel('Membership Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        return membership_values


