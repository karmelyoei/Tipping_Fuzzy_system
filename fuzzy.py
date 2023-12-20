import numpy as np
import matplotlib.pyplot as plt

class Fuzzy:
    def __init__(self):
        pass

    @staticmethod
    def triangular_membership_function(x, a, b, c):
        return np.clip((x - a) / (b - a), 0, 1) * (a < x) * (x <= b) + np.clip((c - x) / (c - b), 0, 1) * (b < x) * (x <= c)

    @staticmethod
    def gaussian_membership_function(x, mean, sigma):
        exponent = -0.5 * ((x - mean) / sigma) ** 2
        return np.exp(exponent)

    @staticmethod
    def trapezoidal_membership_function(x, a, b, c, d):
        return (np.clip((x - a) / (b - a), 0, 1) * (a < x) * (x <= b) +
                1.0 * (b < x) * (x <= c) +
                np.clip((d - x) / (d - c), 0, 1) * (c < x) * (x <= d))

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

# Example usage:
fuzzy_system = Fuzzy()

# Triangular membership function
fuzzy_system.run('triangular', np.linspace(0, 10, 1000), (2, 5, 8))

# Gaussian membership function
fuzzy_system.run('gaussian', np.linspace(0, 10, 1000), (5, 2))

# Trapezoidal membership function
fuzzy_system.run('trapezoidal', np.linspace(0, 10, 1000), (2, 4, 7, 9))
