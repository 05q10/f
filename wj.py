from collections import deque

# -------------------------------------------------------
# BFS APPROACH  (Shortest solution – Complete, Optimal)
# -------------------------------------------------------

def bfs_water_jug(jug1, jug2, target):
    visited = set()
    queue = deque()
    queue.append((0, 0, []))  # (jug1_amount, jug2_amount, path)

    while queue:
        a, b, path = queue.popleft()

        if (a, b) in visited:
            continue
        visited.add((a, b))

        # Goal check
        if a == target or b == target:
            path.append((a, b))
            return path

        # All possible actions
        actions = []

        # 1. Fill Jug1
        actions.append((jug1, b))

        # 2. Fill Jug2
        actions.append((a, jug2))

        # 3. Empty Jug1
        actions.append((0, b))

        # 4. Empty Jug2
        actions.append((a, 0))

        # 5. Pour Jug1 -> Jug2
        transfer = min(a, jug2 - b)
        actions.append((a - transfer, b + transfer))

        # 6. Pour Jug2 -> Jug1
        transfer = min(b, jug1 - a)
        actions.append((a + transfer, b - transfer))

        # Add new states to queue
        for (na, nb) in actions:
            if (na, nb) not in visited:
                queue.append((na, nb, path + [(a, b)]))

    return None


# -------------------------------------------------------
# DFS APPROACH (May not find shortest path)
# -------------------------------------------------------

def dfs_water_jug(jug1, jug2, target):
    visited = set()
    solution = []

    def dfs(a, b, path):
        if (a, b) in visited:
            return False
        visited.add((a, b))

        # Goal
        if a == target or b == target:
            path.append((a, b))
            solution.extend(path)
            return True

        # Allowed actions
        actions = []

        actions.append((jug1, b))  # Fill Jug1
        actions.append((a, jug2))  # Fill Jug2
        actions.append((0, b))     # Empty Jug1
        actions.append((a, 0))     # Empty Jug2

        # Pour Jug1 -> Jug2
        t = min(a, jug2 - b)
        actions.append((a - t, b + t))

        # Pour Jug2 -> Jug1
        t = min(b, jug1 - a)
        actions.append((a + t, b - t))

        # DFS recursion
        for (na, nb) in actions:
            if dfs(na, nb, path + [(a, b)]):
                return True
        return False

    dfs(0, 0, [])
    return solution


# -------------------------------------------------------
# RUN EXAMPLE
# -------------------------------------------------------

if __name__ == "__main__":
    jug1 = 4
    jug2 = 3
    target = 2

    print("\n=== BFS Solution (Shortest Path) ===")
    path = bfs_water_jug(jug1, jug2, target)
    if path:
        for step in path:
            print(step)
    else:
        print("No solution found.")

    print("\n=== DFS Solution ===")
    path = dfs_water_jug(jug1, jug2, target)
    if path:
        for step in path:
            print(step)
    else:
        print("No solution found.")
