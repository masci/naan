from pyparsing import (
    Word,
    alphas,
    alphanums,
    nums,
    Literal,
    QuotedString,
    infixNotation,
    opAssoc,
    ParseException,
)

metadata_key = Word(alphas, alphanums)
integer = Word(nums).setParseAction(lambda t: int(t[0]))
string = QuotedString("'")

operand = metadata_key | integer | string

# Supported operators
EQ = Literal("==")
NE = Literal("!=")
GT = Literal(">")
LT = Literal("<")
GE = Literal(">=")
LE = Literal("<=")

comparison_op = EQ | NE | GT | LT | GE | LE

# Define the structure of a comparison expression
expression = infixNotation(operand, [(comparison_op, 2, opAssoc.LEFT)])


def evaluate_expression(expression, metadata):
    """Function to evaluate a parsed filter expression.

    Assume the expression is in the form [operand, operator, operand]
    """
    if len(expression) != 3:
        raise ValueError("Invalid expression format")

    key, operator, value = expression

    if key not in metadata:
        return False

    metadata_value = metadata[key]

    if operator == "==":
        return metadata_value == value
    elif operator == "!=":
        return metadata_value != value
    elif operator == ">":
        return metadata_value > value
    elif operator == "<":
        return metadata_value < value
    elif operator == ">=":
        return metadata_value >= value
    elif operator == "<=":
        return metadata_value <= value
    else:
        raise ValueError("Unknown operator")
