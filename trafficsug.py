import numpy as np

# ----------------------------
# Membership Functions
# ----------------------------

def tri(x, a, b, c):
    return max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trap(x, a, b, c, d):
    return max(0, min((x - a) / (b - a + 1e-9), 1, (d - x) / (d - c + 1e-9)))


# -----------------------------------------
# Input 1: Traffic Density (cars/min)
# -----------------------------------------

def density_mfs(d):
    return {
        "Low":    tri(d, 0, 10, 20),
        "Medium": tri(d, 10, 25, 40),
        "High":   tri(d, 30, 50, 70),
        "VHigh":  trap(d, 60, 80, 100, 100)
    }


# -----------------------------------------
# Input 2: Waiting Time (seconds)
# -----------------------------------------

def wait_mfs(w):
    return {
        "Short":  tri(w, 0, 10, 20),
        "Medium": tri(w, 15, 30, 45),
        "Long":   trap(w, 40, 70, 110, 140)
    }


# -----------------------------------------
# Sugeno Consequent Constants (Green Time)
# -----------------------------------------

# Output is DIRECT crisp values (Sugeno zero-order)
green_times = {
    "VS": 10,
    "S":  20,
    "M":  30,
    "L":  45,
    "VL": 60
}


# -----------------------------------------
# Sugeno Rule Base
# -----------------------------------------

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
    ("VHigh",  "Long",   "VL")
]


# -----------------------------------------
# Sugeno Inference (Weighted Average)
# -----------------------------------------

def sugeno_infer(density, wait):
    dmf = density_mfs(density)
    wmf = wait_mfs(wait)

    numerator = 0
    denominator = 0

    for (dterm, wterm, out_term) in rules:
        w = min(dmf[dterm], wmf[wterm])      # antecedent firing strength
        z = green_times[out_term]            # consequent constant
        numerator += w * z
        denominator += w

    if denominator == 0:
        return 20  # Default green time

    return numerator / denominator


# -----------------------------------------
# Main Loop
# -----------------------------------------

def main():
    print("\n=== Sugeno Fuzzy Traffic Controller ===\n")

    while True:
        try:
            d = input("Traffic Density (cars/min, default=30): ")
            density = float(d) if d else 30

            w = input("Waiting Time (seconds, default=20): ")
            wait = float(w) if w else 20
        except:
            print("Invalid input! Try again.\n")
            continue

        result = sugeno_infer(density, wait)

        print(f"\n→ Green Time = {result:.2f} seconds (Sugeno Output)\n")

        again = input("Run again? (y/n): ").strip().lower()
        if again == "n":
            break


if __name__ == "__main__":
    main()

