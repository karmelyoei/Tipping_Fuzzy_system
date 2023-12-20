import numpy as np
from control import Control
from fuzzy import Fuzzy


# Build the tipping system
control_system = Control()

# Define input and output linguistic terms and membership functions
service_terms = {'poor': [0, 1.699, 0], 'good': [1.699, 5, 0], 'excellent': [5, 10, 0]}
food_terms = {'rancid': [0, 0, 3, 6], 'delicious': [4, 7, 10, 10]}
tip_terms = {'cheap': [0, 5, 10], 'average': [10, 15, 20], 'generous': [20, 25, 30]}

fuzzy_system = Fuzzy()

service_poor = fuzzy_system.run('gaussian', np.linspace(0, 10, 10), (1.699, 0))
service_good = fuzzy_system.run('gaussian', np.linspace(0, 10, 10), (1.699, 5))
service_excellent = fuzzy_system.run('gaussian', np.linspace(0, 10, 10), (1.699, 10))


food_rancid = fuzzy_system.run('trapezoidal', np.linspace(0, 10, 10), (0, 0, 3, 6))
food_delicious = fuzzy_system.run('trapezoidal', np.linspace(0, 10, 10), (4, 7, 10, 10))


tip_cheap = fuzzy_system.run('triangular', np.linspace(0, 30, 10), (0,5,10))
tip_average = fuzzy_system.run('triangular', np.linspace(0, 30, 10), (10,15,20))
tip_generous = fuzzy_system.run('triangular', np.linspace(0, 30, 10), (20,25,30))


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
control_system.add_rule(['service', 'food'], 'cheap', 'or')
control_system.add_rule(['service'], 'average', None)
control_system.add_rule(['service', 'food'], 'generous', 'or')


# Set input values
input_values = {'service': 6, 'food': 8}

# Compute the result
output_value = control_system.compute(input_values, fuzzy_system)

# Print the result
print("Predicted Tip:", output_value)