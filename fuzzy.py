import numpy as np
import matplotlib.pyplot as plt

class Fuzzy:
    def __init__(self):
        pass

    @staticmethod
    def triangular_membership_function(x, a, b, c):
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    @staticmethod
    def gaussian_membership_function(x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    @staticmethod
    def trapezoidal_membership_function(x, a, b, c, d):
        if x <= a or x >= d:
            # Outside the trapezoid, membership is 0
            return 0.0
        elif a < x <= b:
            # Rising slope
            return (x - a) / (b - a)
        elif b < x <= c:
            # Flat top
            return 1.0
        elif c < x <= d:
            # Falling slope
            return (d - x) / (d - c)
        else:
            # Should not reach here
            return 0.0


    @staticmethod
    def defuzz_centroid(x_values, membership_values, num_points):
        numerator = np.sum(x_values * membership_values * (num_points - 1 + x_values))
        denominator = np.sum(membership_values * (num_points - 1 + x_values))

        if denominator == 0:
            return np.nan  # Handle division by zero

        return numerator / denominator

    def run(self, function_name, x_values, params):
        if function_name == 'triangular':
            membership_values = [self.triangular_membership_function(x, *params) for x in x_values]
            title = 'Triangular Membership Function'
        elif function_name == 'gaussian':
            membership_values = [self.gaussian_membership_function(x, *params) for x in x_values]
            title = 'Gaussian Membership Function'
        elif function_name == 'trapezoidal':
            membership_values = [self.trapezoidal_membership_function(x, *params) for x in x_values]
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


