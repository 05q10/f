import numpy as np

# ------------------------
# Membership functions
# ------------------------

def tri(x, a, b, c):
    return max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trap(x, a, b, c, d):
    return max(0, min((x - a) / (b - a + 1e-9), 1, (d - x) / (d - c + 1e-9)))


# ------------------------------------------------
# 1. Fuzzy sets for INPUT 1: Traffic Density
# ------------------------------------------------

def density_mfs(d):
    return {
        "Low":    tri(d, 0, 10, 20),
        "Medium": tri(d, 10, 25, 40),
        "High":   tri(d, 30, 50, 70),
        "VHigh":  trap(d, 60, 80, 100, 100)
    }


# ------------------------------------------------
# 2. Fuzzy sets for INPUT 2: Waiting Time
# ------------------------------------------------

def wait_mfs(w):
    return {
        "Short":  tri(w, 0, 10, 20),
        "Medium": tri(w, 10, 25, 40),
        "Long":   trap(w, 30, 60, 90, 120)
    }


# ------------------------------------------------
# 3. Fuzzy sets for OUTPUT: Green Time (seconds)
# ------------------------------------------------

green_time_vals = {
    "VS": 5,
    "S":  15,
    "M":  25,
    "L":  40,
    "VL": 60
}


# ------------------------------------------------
# Rule base: (Density, WaitingTime → GreenTime)
# ------------------------------------------------

rules = [
    ("Low",    "Short",  "VS"),
    ("Low",    "Medium", "S"),
    ("Low",    "Long",   "M"),

    ("Medium", "Short",  "S"),
    ("Medium", "Medium", "M"),
    ("Medium", "Long",   "L"),

    ("High",   "Short",  "M"),
    ("High",   "Medium", "L"),
    ("High",   "Long",   "VL"),

    ("VHigh",  "Short",  "L"),
    ("VHigh",  "Medium", "VL"),
    ("VHigh",  "Long",   "VL"),
]


# ------------------------------------------------
# Mamdani Inference (min rule)
# THEN part uses singleton membership (Sugeno-like)
# Final step: Weighted average defuzzification
# ------------------------------------------------

def fuzzy_infer(density, wait):
    dmf = density_mfs(density)
    wmf = wait_mfs(wait)

    num = 0
    den = 0

    for (dterm, wterm, gterm) in rules:
        alpha = min(dmf[dterm], wmf[wterm])  # firing strength
        num += alpha * green_time_vals[gterm]
        den += alpha

    if den == 0:
        return 20   # default green time

    return num / den


# ------------------------------------------------
# SIMULATION LOOP (multiple inputs)
# ------------------------------------------------

def main():
    print("\n=== Fuzzy Traffic Signal Controller ===\n")
    print("Enter traffic conditions. Press Enter for default values.\n")

    while True:
        try:
            d = input("Traffic Density (cars/min, default=30): ").strip()
            density = float(d) if d else 30

            w = input("Waiting Time (seconds, default=20): ").strip()
            wait = float(w) if w else 20

        except:
            print("Invalid input! Try again.\n")
            continue

        gt = fuzzy_infer(density, wait)

        print(f"\n→ RECOMMENDED GREEN TIME = {gt:.2f} seconds\n")

        again = input("Run again? (y/n): ").strip().lower()
        if again == "n":
            break


if __name__ == "__main__":
    main()

