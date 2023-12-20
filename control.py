import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class Control:
    def __init__(self):
        self.antecedents = {}
        self.consequent = None
        self.rules = []
        self.input_values = {}

    def add_antecedent(self, name, universe, linguistic_terms):
        self.antecedents[name] = fuzz.Antecedent(universe, name)
        for term, params in linguistic_terms.items():
            self.antecedents[name][term] = fuzz.gaussmf(universe, *params)

    def add_consequent(self, name, universe, linguistic_terms):
        self.consequent = fuzz.Consequent(universe, name)
        for term, params in linguistic_terms.items():
            self.consequent[term] = fuzz.gaussmf(universe, *params)

    @staticmethod
    def fuzzy_or(membership_values_list):
        return np.maximum.reduce(membership_values_list)

    @staticmethod
    def fuzzy_and(membership_values_list):
        return np.minimum.reduce(membership_values_list)

    @staticmethod
    def fuzzy_not(membership_values):
        return 1 - membership_values

    def add_rule(self, antecedents, consequent, operator='or'):
        self.rules.append({'antecedents': antecedents, 'consequent': consequent, 'operator': operator})

    def evaluate_rule(self, antecedents, operator='or'):
        if operator == 'or':
            return self.fuzzy_or(antecedents)
        elif operator == 'and':
            return self.fuzzy_and(antecedents)
        elif operator == 'not':
            return self.fuzzy_not(antecedents)
        else:
            raise ValueError("Invalid operator. Use 'or', 'and', or 'not'.")

    def aggregate_rules(self, x_values):
        aggregated_result = np.zeros_like(x_values, dtype=float)

        for rule in self.rules:
            antecedent_values = [
                np.interp(x_values, np.arange(len(self.antecedents[antecedent_name][term])), self.antecedents[antecedent_name][term])
                for antecedent_name, term in rule['antecedents']
            ]
            rule_result = self.evaluate_rule(antecedent_values, rule['operator'])
            aggregated_result = np.maximum(aggregated_result, rule_result)

        return aggregated_result

    def compute(self):
        if not self.input_values or not self.consequent:
            raise ValueError("Input values or consequent not set.")

        x_values = np.arange(len(self.consequent.universe))
        aggregated_result = self.aggregate_rules(x_values)

        # Compute the result using a simple centroid defuzzification
        output_value = fuzz.defuzz(x_values, aggregated_result, 'centroid')
        self.consequent['output'] = output_value

        return output_value


