import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Input variables
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
tip = ctrl.Consequent(np.arange(0, 31, 1), 'tip')

# Membership functions
service['poor'] = fuzz.gaussmf(service.universe, 0, 1.69)
service['good'] = fuzz.gaussmf(service.universe, 5, 1.699)
service['excellent'] = fuzz.gaussmf(service.universe, 10, 1.69)

food['rancid'] = fuzz.trapmf(food.universe, [0, 0, 3, 6])
food['delicious'] = fuzz.trapmf(food.universe, [4, 7, 10, 10])

tip['cheap'] = fuzz.trimf(tip.universe, [0, 5, 10])
tip['average'] = fuzz.trimf(tip.universe, [10, 15, 20])
tip['generous'] = fuzz.trimf(tip.universe, [20, 25, 30])

# Rules
rule1 = ctrl.Rule(service['poor'] | food['rancid'], tip['cheap'])
rule2 = ctrl.Rule(service['good'], tip['average'])
rule3 = ctrl.Rule(service['excellent'] | food['delicious'], tip['generous'])

# Fuzzy system
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Example input
tipping.input['service'] = 3
tipping.input['food'] = 8

# Compute the result
tipping.compute()

# Print the result
print("Tip:", tipping.output['tip'])

# Plot the membership functions and the input/output
service.view()
food.view()
tip.view(sim=tipping)
plt.show()
