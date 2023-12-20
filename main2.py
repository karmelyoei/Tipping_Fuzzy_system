import numpy as np
from control import Control
from fuzzy import Fuzzy



# Build the tipping system
control_system = Control()

# Define input and output linguistic terms and membership functions
service_terms = {'poor': [0, 1.699, 0], 'good': [1.699, 5, 0], 'excellent': [5, 10, 0]}
food_terms = {'rancid': [0, 0, 3, 6], 'delicious': [4, 7, 10, 10]}
tip_terms = {'cheap': [0, 5, 10], 'average': [10, 15, 20], 'generous': [20, 25, 30]}

# Add antecedents (input variables) and consequent (output variable) to the control system
control_system.add_antecedent('service', np.arange(0, 11, 1), service_terms['poor'])
control_system.add_antecedent('food', np.arange(0, 11, 1), food_terms['rancid'])
control_system.add_consequent('tip', np.arange(0, 31, 1), tip_terms['cheap'])

# Add fuzzy rules to the control system
control_system.add_rule(['service', 'food'], 'cheap', 'or')

# Set input values
input_values = {'service': 6, 'food': 8}

# Compute the result
output_value = control_system.compute(input_values)

# Print the result
print("Predicted Tip:", output_value)