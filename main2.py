import numpy as np
from control import Control
from fuzzy import Fuzzy
import matplotlib.pyplot as plt

# Build the tipping system
control_system = Control()

fuzzy_system = Fuzzy()

service_poor = fuzzy_system.run('gaussian', np.linspace(0, 10, 10), (0, 1.699))
service_good = fuzzy_system.run('gaussian', np.linspace(0, 10, 10), (5, 1.699))
service_excellent = fuzzy_system.run('gaussian', np.linspace(0, 10, 10), (10, 1.699))

# Plot all fuzzy sets on the same graph
plt.plot(np.linspace(0, 10, 10), service_poor, label='Service Poor')
plt.plot(np.linspace(0, 10, 10), service_good, label='Service Good')
plt.plot(np.linspace(0, 10, 10), service_excellent, label='Service Excellent')

# Add labels and legend
plt.xlabel('X-axis Label')
plt.ylabel('Membership Value')
plt.title('Fuzzy Sets for Service')
plt.legend()

# Show the plot
plt.show()

food_rancid = fuzzy_system.run('trapezoidal', np.linspace(0, 10, 10), (0, 0, 3, 6))
food_delicious = fuzzy_system.run('trapezoidal', np.linspace(0, 10, 10), (4, 7, 10, 10))

# Plot only food fuzzy sets on the same graph
plt.plot(np.linspace(0, 10, 10), food_rancid, label='Food Rancid')
plt.plot(np.linspace(0, 10, 10), food_delicious, label='Food Delicious')

# Add labels and legend
plt.xlabel('X-axis Label')
plt.ylabel('Membership Value')
plt.title('Fuzzy Sets for Food')
plt.legend()

# Show the plot
plt.show()

tip_cheap = fuzzy_system.run('triangular', np.linspace(0, 30, 10), (0,5,10))
tip_average = fuzzy_system.run('triangular', np.linspace(0, 30, 10), (10,15,20))
tip_generous = fuzzy_system.run('triangular', np.linspace(0, 30, 10), (20,25,30))

tip_universe = np.linspace(0, 30, 10)
# Plot the membership functions
plt.plot(tip_universe, tip_cheap, label='Cheap')
plt.plot(tip_universe, tip_average, label='Average')
plt.plot(tip_universe, tip_generous, label='Generous')

# Add labels and legend
plt.xlabel('Tip Value')
plt.ylabel('Membership Value')
plt.title('Fuzzy Sets for Tip')
plt.legend()

# Show the plot
plt.show()

# Add antecedents (input variables) and consequent (output variable) to the control system
control_system.add_antecedent('service','poor', service_poor )
control_system.add_antecedent('service', 'good',service_good )
control_system.add_antecedent('service', 'excellent',service_excellent )


control_system.add_antecedent('food', 'rancid', food_rancid)
control_system.add_antecedent('food', 'delicious', food_delicious)


control_system.add_consequent('tip','cheap', tip_cheap)
control_system.add_consequent('tip','average', tip_average)
control_system.add_consequent('tip','generous', tip_generous)


# Add fuzzy rules to the control system
# Rule 1: if service is poor or food is rancid, then tip is cheap
control_system.add_rule((['service', 'poor'], ['food', 'rancid']), 'cheap', 'or')
# Rule 2: if service is good, then tip is average
control_system.add_rule((['service', 'good'], [None,None]), 'average', None)
# Rule 3: if service is excellent or food is delicious, then tip is generous
control_system.add_rule((['service', 'excellent'], ['food', 'delicious']), 'generous', 'or')


# Set input values
input_values = {'service': 6, 'food': 8}

# Compute the result
output_value = control_system.compute(input_values, fuzzy_system)

# Print the result
print("Predicted Tip:", output_value)