import operator as op

ops = {
        '>': op.gt,
        '<': op.lt,
        '>=': op.ge,
        '<=': op.le,
        '==': op.eq,
        '!=': op.ne
    }


class Predicate:
    """
    Predicate class to represent a predicate consisting of a feature, operator and value.
    """

    def __init__(self, feature: str, operator: str, value: float):
        """
        Initialize the Predicate object.

        Args:
            feature (str): The feature name.
            operator (str): The operator (e.g., '>', '<', '==', etc.).
            value (float): The value to compare against.
        """
        self.feature = feature
        self.operator = operator
        self.value = value

    def evaluate(self, row):
        return ops[self.operator](row[self.feature], self.value)

    def __repr__(self):
        return f"Predicate({self.feature} {self.operator} {self.value})"
