import numpy as np

# -------------------------------------------
# Given discrete membership functions
# -------------------------------------------

# (value, membership)
tall = {
    5: 0.2,
    7: 0.3,
    9: 0.7,
    11: 0.9,
    12: 1.0
}

short = {
    0: 0.3,
    30: 0.0,
    60: 1.0,
    90: 0.5,
    120: 0.0
}

# -------------------------------------------
# Linguistic Modifiers
# -------------------------------------------

def very(mu):
    """ Intensifier: very A = (mu)^2 """
    return mu ** 2

def fairly(mu):
    """ Diluter: fairly A = sqrt(mu) """
    return np.sqrt(mu)

def not_(mu):
    """ Negation: not A = 1 - mu """
    return 1 - mu


# -------------------------------------------
# Compute new membership functions
# -------------------------------------------

very_tall = {x: very(mu) for x, mu in tall.items()}
fairly_tall = {x: fairly(mu) for x, mu in tall.items()}

very_short = {x: very(mu) for x, mu in short.items()}
not_very_short = {x: not_(very_short[x]) for x in short.keys()}


# -------------------------------------------
# Print results in NICE formatting
# -------------------------------------------

print("\nVERY TALL Membership Function:")
for x, mu in very_tall.items():
    print(f"x={x}, μ={mu:.3f}")

print("\nFAIRLY TALL Membership Function:")
for x, mu in fairly_tall.items():
    print(f"x={x}, μ={mu:.3f}")

print("\nNOT VERY SHORT Membership Function:")
for x, mu in not_very_short.items():
    print(f"x={x}, μ={mu:.3f}")
