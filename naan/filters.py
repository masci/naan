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
