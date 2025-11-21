import random
import copy

# ----------------------------------------------------
# Performance Metrics Container
# ----------------------------------------------------
class Metrics:
    def __init__(self):
        self.cleaned = 0
        self.energy = 0
        self.steps = 0
        self.moves = 0
        self.sucks = 0

    def score(self):
        # Higher cleaned, fewer energy/steps = better score
        return self.cleaned - 0.1*self.energy


# ----------------------------------------------------
# Rule-Based Sequential Cleaning Agent
# ----------------------------------------------------
def sequential_cleaner(env):
    env = copy.deepcopy(env)
    n = len(env)
    metrics = Metrics()

    for i in range(n):
        metrics.steps += 1
        if env[i] == 1:
            env[i] = 0
            metrics.cleaned += 1
            metrics.sucks += 1
            metrics.energy += 2   # energy for suction
        metrics.energy += 1       # energy for moving/idle

    return env, metrics


# ----------------------------------------------------
# Random Cleaning Agent
# ----------------------------------------------------
def random_cleaner(env, steps=30):
    env = copy.deepcopy(env)
    n = len(env)
    pos = random.randint(0, n-1)  # random start position

    metrics = Metrics()

    for _ in range(steps):
        metrics.steps += 1

        action = random.choice(["LEFT", "RIGHT", "SUCK"])

        if action == "SUCK":
            metrics.sucks += 1
            metrics.energy += 2
            if env[pos] == 1:
                env[pos] = 0
                metrics.cleaned += 1

        elif action == "LEFT":
            metrics.moves += 1
            metrics.energy += 1
            if pos > 0:
                pos -= 1

        elif action == "RIGHT":
            metrics.moves += 1
            metrics.energy += 1
            if pos < n-1:
                pos += 1

    return env, metrics


# ----------------------------------------------------
# Comparison and Main Execution
# ----------------------------------------------------
def main():
    # Example environment input
    environment = [1, 0, 1, 1, 0, 1]
    print("Environment:", environment)

    # Run Sequential Cleaner
    seq_env, seq_met = sequential_cleaner(environment)
    print("\n--- Sequential Cleaner ---")
    print("Final:", seq_env)
    print(f"Cleaned: {seq_met.cleaned}, Steps: {seq_met.steps}, Energy: {seq_met.energy}")

    # Run Random Cleaner
    rand_env, rand_met = random_cleaner(environment, steps=30)
    print("\n--- Random Cleaner ---")
    print("Final:", rand_env)
    print(f"Cleaned: {rand_met.cleaned}, Steps: {rand_met.steps}, Energy: {rand_met.energy}")

    # Compare Using Score
    seq_score = seq_met.score()
    rand_score = rand_met.score()

    print("\n--- Performance Comparison (Higher is better) ---")
    print(f"Sequential Score: {seq_score:.2f}")
    print(f"Random Score:     {rand_score:.2f}")

    if seq_score > rand_score:
        print("\n→ BEST AGENT: SEQUENTIAL (Rule-Based)")
    else:
        print("\n→ BEST AGENT: RANDOM (Rare, only if lucky)")


if __name__ == "__main__":
    main()
