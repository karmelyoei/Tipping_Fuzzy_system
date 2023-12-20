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

        # Aggregate rules
        x_values = np.arange(len(next(iter(self.consequent.values()))))
        aggregated_result = self.aggregate_rules(x_values)

        # Compute the result using a simple centroid defuzzification
        output_value = fuzz.defuzz_centroid(x_values, aggregated_result, len(aggregated_result))

        return output_value


