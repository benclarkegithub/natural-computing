import math
from GeneticProgram import Function

# Functions
plus = Function(lambda x: x[0] + x[1], 2, None, None, "(", "+", ")")
minus = Function(lambda x: x[0] - x[1], 2, None, None, "(", "-", ")")
multiply = Function(lambda x: x[0] * x[1], 2, None, None, "(", "*", ")")
divide = Function(lambda x: x[0] / x[1], 2, None, lambda x: math.isclose(x[1], 0), "(", "/", ")")
sin = Function(lambda x: math.sin(x[0]), 1, None, None, "sin(", "", ")")
cos = Function(lambda x: math.cos(x[0]), 1, None, None, "cos(", "", ")")
sin_pi = Function(lambda x: math.sin(math.pi * x[0]), 1, None, None, "sin(π*", "", ")")
cos_pi = Function(lambda x: math.cos(math.pi * x[0]), 1, None, None, "cos(π*", "", ")")
exp = Function(lambda x: math.exp(x[0]), 1, None, lambda x: x[0] >= 10, "exp(", "", ")")
absolute = Function(lambda x: abs(x[0]), 1, None, None, "abs(", "", ")")
power_2 = Function(lambda x: x[0] ** 2, 1, None, None, "", "", "^2")
power_3 = Function(lambda x: x[0] ** 3, 1, None, None, "", "", "^3")
power_4 = Function(lambda x: x[0] ** 4, 1, None, None, "", "", "^4")
power_1_2 = Function(lambda x: math.sqrt(x[0]), 1, None, lambda x: x[0] < 0, "", "", "^(1/2)")
power_1_3 = Function(lambda x: (x[0] ** (1/3)) if x[0] >= 0 else -(abs(x[0]) ** (1/3)), 1, None, None, "", "", "^(1/3)")
power_1_4 = Function(lambda x: x[0] ** 0.25, 1, None, lambda x: x[0] < 0, "", "", "^(1/4)")

# These values are just used for probabilities
div_1_14 = 1 / 14
div_1_30 = 1 / 30
function_set = [
    { "function": plus, "probability": 0.1 },
    { "function": minus, "probability": 0.1 },
    { "function": multiply, "probability": 0.1, "disallowed_terminals": [1] },
    { "function": divide, "probability": 0.1 },
    {
        "function": sin,
        "probability": 0.02,
        "disallowed_functions": [sin, cos, sin_pi, cos_pi, exp],
        "disallowed_terminals": [1, 2, 3, 4, 10, 100]
    },
    {
        "function": cos,
        "probability": 0.02,
        "disallowed_functions": [sin, cos, sin_pi, cos_pi, exp],
        "disallowed_terminals": [1, 2, 3, 4, 10, 100]
    },
    {
        "function": sin_pi,
        "probability": 0.08,
        "disallowed_functions": [sin, cos, sin_pi, cos_pi, exp],
        "disallowed_terminals": [1, 2, 3, 4, 10, 100]
    },
    {
        "function": cos_pi,
        "probability": 0.08,
        "disallowed_functions": [sin, cos, sin_pi, cos_pi, exp],
        "disallowed_terminals": [1, 2, 3, 4, 10, 100]
    },
    {
        "function": exp,
        "probability": 0.1,
        "disallowed_functions": [sin, cos, sin_pi, cos_pi, exp],
        "disallowed_terminals": [10, 100]
    },
    {
        "function": absolute,
        "probability": 0.1,
        "disallowed_functions": [sin, cos, sin_pi, cos_pi, exp, absolute],
        "disallowed_terminals": [1, 2, 3, 4, 10, 100, math.pi]
    },
    {
        "function": power_2,
        "probability": div_1_30,
        "disallowed_functions": [absolute, exp, power_2, power_3, power_4, power_1_2, power_1_3, power_1_4],
        "disallowed_terminals": [10, 100]
    },
    {
        "function": power_3,
        "probability": div_1_30,
        "disallowed_functions": [exp, power_2, power_3, power_4, power_1_2, power_1_3, power_1_4],
        "disallowed_terminals": [10, 100]
    },
    {
        "function": power_4,
        "probability": div_1_30,
        "disallowed_functions": [absolute, exp, power_2, power_3, power_4, power_1_2, power_1_3, power_1_4],
        "disallowed_terminals": [10, 100]
    },
    {
        "function": power_1_2,
        "probability": div_1_30,
        "disallowed_functions": [exp, power_2, power_3, power_4, power_1_2, power_1_3, power_1_4],
        "disallowed_terminals": [10, 100]
    },
    {
        "function": power_1_3,
        "probability": div_1_30,
        "disallowed_functions": [exp, power_2, power_3, power_4, power_1_2, power_1_3, power_1_4],
        "disallowed_terminals": [10, 100]
    },
    {
        "function": power_1_4,
        "probability": div_1_30,
        "disallowed_functions": [exp, power_2, power_3, power_4, power_1_2, power_1_3, power_1_4],
        "disallowed_terminals": [10, 100]
    }
]

terminal_set = [
    { "value": 1, "probability": div_1_14 },
    { "value": 2, "probability": div_1_14 },
    { "value": 3, "probability": div_1_14 },
    { "value": 4, "probability": div_1_14 },
    { "value": 10, "probability": div_1_14 },
    { "value": 100, "probability": div_1_14 },
    { "value": math.pi, "probability": div_1_14 },
    { "value": 'x', "probability": 0.25 },
    { "value": 'y', "probability": 0.25 }
]
