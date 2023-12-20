import numpy as np
import matplotlib.pyplot as plt

# Define membership functions
def triangular_mf(x, a, b, c):
    """
    Triangular membership function.

    Parameters:
    - x: Input value
    - a: Left point
    - b: Peak point
    - c: Right point

    Returns:
    - Degree of membership for x in the triangular fuzzy set
    """
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return (c - x) / (c - b)
    else:
        return 0

def gaussian_mf(x, mu, sigma):
    """
    Gaussian membership function.

    Parameters:
    - x: Input value
    - mu: Mean
    - sigma: Standard deviation

    Returns:
    - Degree of membership for x in the Gaussian fuzzy set
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def trapezoidal_mf(x, a, b, c, d):
    """
    Trapezoidal membership function.

    Parameters:
    - x: Input value
    - a: Starting point
    - b: Left slope
    - c: Right slope
    - d: Ending point

    Returns:
    - Degree of membership for x in the trapezoidal fuzzy set
    """
    if x <= a or x >= d:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    else:
        return 0




# Create a 2D grid of input values
# service_grid, food_grid = np.meshgrid(service_values, food_values)

# Define fuzzy sets
# Define input variables
service_values = np.arange(0, 11, 0.1)
# Calculate the membership values using the Gaussian membership function
service_poor = [gaussian_mf(x, 0, 1.699) for x in service_values]
service_good= [gaussian_mf(x, 5, 1.699) for x in service_values]
service_excellent = [gaussian_mf(x, 10, 1.699) for x in service_values]

# Plotting the Gaussian membership function
plt.plot(service_values, service_poor, label='Poor')
plt.plot(service_values, service_good, label='Good')
plt.plot(service_values, service_excellent, label='Excellent')
plt.title('Gaussian Membership Function For Service')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()

# Create the second set of values for delicious food
food_values = np.arange(0, 11, 0.1)
a, b, c, d = 0,0,3,6
# Calculate the membership values for each X value
food_rancid = [trapezoidal_mf(x, a, b, c, d) for x in food_values]

# Calculate the membership values for the second set of X values
# You can use the same trapezoidal_mf function and adjust the parameters as needed
a2, b2, c2, d2 = 4,7,10,10
food_delicious = [trapezoidal_mf(x, a2, b2, c2, d2) for x in food_values]

# Plotting both sets of membership functions
plt.plot(food_values, food_rancid, label='Food Rancid')
plt.plot(food_values, food_delicious, label='Food Delicious')
plt.title('Trapezoidal Membership Functions For Food')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()


#  Define output variable Tipping
tip_values = np.arange(0, 31, 0.1)

# Calculate the membership values using the triangular membership function
tip_cheap = [triangular_mf(x, 0, 5, 10) for x in tip_values]
tip_average = [triangular_mf(x, 10, 15, 20) for x in tip_values]
tip_generous = [triangular_mf(x, 20, 25, 30) for x in tip_values]

# Plotting the triangular membership function
plt.plot(tip_values, tip_cheap, label='Cheap')
plt.plot(tip_values, tip_average, label='Average')
plt.plot(tip_values, tip_generous, label='Generous')
plt.title('Triangular Membership Function For Tipping')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()


# Calculate the activation of the rule: service is poor OR food is rancid
# Rule 1: if service is poor or food is rancid, then tip is cheap
rule_activation = np.fmin(service_poor, food_rancid)
tip_activation_cheap = np.fmin(rule_activation, tip_cheap[:rule_activation.shape[0]])

# Plot the membership functions and the rule activation
plt.plot(range(110), service_poor, label='Service Membership')
plt.plot(range(110), food_rancid, label='Food Membership')
plt.plot(range(110), tip_cheap[:rule_activation.shape[0]], label='Tip Membership')
plt.plot(range(110), rule_activation, label='Rule Activation')
plt.plot(range(110), tip_activation_cheap, label='Resulting Tip Membership')
plt.title('Fuzzy Logic: Rule Application (1)')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()


# Rule 2: if service is good, then tip is average
rule2 = service_good
tip_activation_average = np.fmin(rule2, tip_average[:len(service_good)])

plt.plot(range(110), service_good, label='Service Membership (Good)')
plt.plot(range(110), tip_average[:len(service_good)], label='Tip Membership (Average)')
plt.plot(range(110), tip_activation_average, label='Resulting Tip Membership')
plt.title('Fuzzy Logic: Rule Application (2)')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()


# Rule 3: if service is excellent or food is delicious, then tip is generous
rule3 = np.fmin(service_excellent, food_delicious)
tip_activation_generous = np.fmin(rule3, tip_generous[:rule3.shape[0]])

# Aggregate all the output membership functions
aggregated = np.fmax(tip_activation_cheap, np.fmax(tip_activation_average, tip_activation_generous))

# Defuzzify (centroid method)
tip_result = np.sum(tip_values[:aggregated.shape[0]] * aggregated) / np.sum(aggregated)

# Visualize aggregated membership function
plt.plot(tip_values[:aggregated.shape[0]], aggregated, label='Aggregated')
plt.title('Aggregated Output Membership Function')
plt.legend()
plt.show()

# Print the final result
print("Tip:", tip_result)


##############
#Testing
# Choose specific values for service and food
service_value = 6
food_value = 8

# Evaluate the membership functions for the chosen service and food values
service_membership_value = gaussian_mf(service_value, 5, 1.699)
food_membership_value_rancid = trapezoidal_mf(food_value, 0, 0, 3, 6)
food_membership_value_delicious = trapezoidal_mf(food_value, 4, 7, 10, 10)

# Apply the fuzzy rules
# Rule 1: if service is poor or food is rancid, then tip is cheap
rule_activation_1 = np.fmin(service_membership_value, food_membership_value_rancid)
tip_activation_cheap = np.fmin(rule_activation_1, tip_cheap)

# Rule 2: if service is good, then tip is average
rule_activation_2 = service_good[:len(service_values)]  # Use the predefined membership function for "good service"
tip_activation_average = np.fmin(rule_activation_2, tip_average)

# Rule 3: if service is excellent or food is delicious, then tip is generous
rule_activation_3 = np.fmin(service_membership_value, food_membership_value_delicious)
tip_activation_generous = np.fmin(rule_activation_3, tip_generous)

# Aggregate the membership values
aggregated = np.fmax(tip_activation_cheap, np.fmax(tip_activation_average, tip_activation_generous))

# Defuzzify (centroid method)
tip_result = np.sum(tip_values * aggregated) / np.sum(aggregated)

# Visualize the input and output
plt.plot(service_values, service_poor, label='Service Membership (Poor)')
plt.plot(service_values, service_good, label='Service Membership (Good)')
plt.plot(service_values, service_excellent, label='Service Membership (Excellent)')
plt.plot([service_value], [service_membership_value], 'ro', label='Chosen Service Value')

plt.plot(food_values, food_rancid, label='Food Membership (Rancid)')
plt.plot(food_values, food_delicious, label='Food Membership (Delicious)')
plt.plot([food_value], [food_membership_value_rancid], 'ro', label='Chosen Food Value')

plt.plot(tip_values, tip_cheap, label='Tip Membership (Cheap)')
plt.plot(tip_values, tip_average, label='Tip Membership (Average)')
plt.plot(tip_values, tip_generous, label='Tip Membership (Generous)')

plt.plot(tip_values, aggregated, label='Aggregated Output')

plt.title('Fuzzy Logic: Rule Application and Defuzzification')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()

# Print the final result
print("Chosen Service Value:", service_value)
print("Chosen Food Value:", food_value)
print("Tip Result:", tip_result)