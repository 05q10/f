import numpy as np

# ----------- Membership Functions -----------

def trap(x, a, b, c, d):
    return max(0, min((x - a) / (b - a + 1e-9), 1, (d - x) / (d - c + 1e-9)))


def tri(x, a, b, c):
    return max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))


# ----------- Fuzzy Variables -----------

def height_mfs(h):
    return {
        "VH": trap(h, 700, 900, 1000, 1100),     # Very High
        "H":  tri(h, 400, 600, 800),             # High
        "M":  tri(h, 200, 400, 600),             # Medium
        "L":  tri(h, 50, 150, 300),              # Low
        "VL": trap(h, 0, 0, 50, 100)             # Very Low
    }


def velocity_mfs(v):
    return {
        "FD": trap(v, -40, -30, -20, -10),       # Fast Down
        "MD": tri(v, -25, -15, -5),              # Medium Down
        "SD": tri(v, -10, -5, 0),                # Slow Down
        "Z":  tri(v, -1, 0, 1),                  # Zero
        "UP": trap(v, 0, 5, 15, 20)              # Upward
    }


def force_output_levels():
    return {
        "PD": -10,      # Push Down
        "ZF": 0,        # Zero Force
        "SU": +5,       # Small Up
        "MU": +10,      # Medium Up
        "STU": +20      # Strong Up
    }


# ------------- Fuzzy Rule Base --------------

rules = [
    ("VH", "FD", "PD"),
    ("VH", "MD", "PD"),
    ("VH", "SD", "ZF"),

    ("H", "FD", "PD"),
    ("H", "MD", "ZF"),
    ("H", "SD", "SU"),

    ("M", "FD", "ZF"),
    ("M", "MD", "SU"),
    ("M", "SD", "MU"),
    ("M", "Z",  "MU"),

    ("L", "FD", "STU"),
    ("L", "MD", "STU"),
    ("L", "SD", "STU"),
    ("L", "Z",  "STU"),

    ("VL", "FD", "STU"),
    ("VL", "MD", "STU"),
    ("VL", "SD", "STU"),
    ("VL", "Z",  "STU")
]


# ---------- Mamdani Inference ------------

def fuzzy_infer(h, v):
    hmf = height_mfs(h)
    vmf = velocity_mfs(v)
    f_levels = force_output_levels()

    force_sum = 0
    weight_sum = 0

    for hterm, vterm, fterm in rules:
        w = min(hmf[hterm], vmf[vterm])     # AND condition
        force_sum += w * f_levels[fterm]
        weight_sum += w

    if weight_sum == 0:
        return 0

    return force_sum / weight_sum


# -------------- Simulation -----------------

def simulate():
    h = 900     # feet
    v = -20     # ft/s downward
    cycle = 0

    print("\n--- Aircraft Landing Simulation ---\n")

    while h > 0:
        f = fuzzy_infer(h, v)       # compute control force
        v = v + f                   # update velocity
        h = h + v                   # update height
        cycle += 1

        print(f"Cycle {cycle:3d}:  h={h:8.2f} ft   v={v:8.2f} ft/s   f={f:6.2f}")

    print("\nAircraft has landed.\n")

# Run
simulate()
