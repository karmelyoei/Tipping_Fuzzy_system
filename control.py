import numpy as np


class Control:
    def __init__(self):
        self.antecedents = {}
        self.consequent = {}
        self.rules = []

    def add_antecedent(self, name, term, universe):
        if name not in self.antecedents:
            self.antecedents[name] = {}
        self.antecedents[name][term] = universe

    def add_consequent(self, name,term, universe):
            if term not in self.consequent:
                self.consequent[term] = []
            if universe is not None:
                self.consequent[term] = universe
            else:
                print(name, "has None value")

    @staticmethod
    def fuzzy_or(membership_values_list):
        return np.maximum.reduce(membership_values_list)

    @staticmethod
    def fuzzy_and(membership_values_list):
        return np.minimum.reduce(membership_values_list)

    @staticmethod
    def fuzzy_not(membership_values):
        return 1 - membership_values

    @staticmethod
    def fuzzy_none(membership_values):
       return membership_values


    def add_rule(self, antecedents, consequent, operator='or'):
        self.rules.append({'antecedents': antecedents, 'consequent': consequent, 'operator': operator})

    def evaluate_rule(self, antecedents, operator='or'):
        if operator == 'or':
            return self.fuzzy_or(antecedents)
        elif operator == 'and':
            return self.fuzzy_and(antecedents)
        elif operator == 'not':
            return self.fuzzy_not(antecedents)
        elif operator == None:
            return self.fuzzy_none(antecedents)
        else:
            raise ValueError("Invalid operator. Use 'or', 'and', or 'not'.")

    def aggregate_rules(self, x_values):
        aggregated_result = np.zeros_like(x_values, dtype=float)

        for rule in self.rules:
            rule_result = np.ones_like(x_values, dtype=float)  # Initialize with ones for MAX operation

            for antecedent_name, term in rule['antecedents']:
                if antecedent_name is None or term is None:
                    continue

                antecedent_values = np.interp(
                    x_values,
                    np.arange(len(self.antecedents[antecedent_name][term])),
                    self.antecedents[antecedent_name][term]
                )
                rule_result = np.minimum(rule_result, antecedent_values)  # Use MIN for AND-like connectives

            aggregated_result = np.maximum(aggregated_result, rule_result)  # Use MAX for OR-like connectives

        return aggregated_result

    def compute(self, input_values, fuzz, num_points):
        if not input_values or not self.consequent:
            raise ValueError("Input values or consequent not set.")

        service_poor,service_good,service_excellent,food_rancid,food_delicious = 0,0,0,0,0
        for key, value in input_values.items():
            if key == "service":
                # Evaluate the membership functions for the chosen service and food values
                service_poor = fuzz.gaussian_membership_function(value, 0, 1.699)
                service_good = fuzz.gaussian_membership_function(value, 5, 1.699)
                service_excellent = fuzz.gaussian_membership_function(value, 10, 1.699)

            elif key == "food":
                food_rancid = fuzz.trapezoidal_membership_function(value, 0, 0, 3, 6)
                food_delicious = fuzz.trapezoidal_membership_function(value, 4, 7, 10, 10)

        # Apply the fuzzy rules
        # Rule 1: if service is poor or food is rancid, then tip is cheap
        tip_cheap = np.fmax(service_poor, food_rancid)

        # Rule 2: if service is good, then tip is average
        tip_average = service_good

        # Rule 3: if service is excellent or food is delicious, then tip is generous
        tip_generous = np.fmax(service_excellent, food_delicious)

        tip_cheap_center = (0+ 10) / 2
        tip_average_center = (10 + 20) / 2
        tip_generous_center = (20 + 30) / 2

        # Calculate the center of gravity
        sample_cheap = tip_cheap_center - 1
        sample2_cheap = tip_cheap_center + 1
        sample1_average = tip_average_center -1
        sample2_average = tip_average_center + 1
        sample1_generous = tip_generous_center -1
        sample2_generous = tip_generous_center +1

        center_of_gravity = (((sample_cheap + sample2_cheap + tip_cheap_center) * tip_cheap) + ((sample1_generous + sample2_generous + tip_generous_center) * tip_generous) + ((sample1_average + sample2_average + tip_average_center) * tip_average)) / ((3 * tip_average) + (3 * tip_generous) + (3 * tip_cheap))

        return center_of_gravity


