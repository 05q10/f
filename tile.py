import numpy as np
from queue import PriorityQueue, Queue

# Helper functions
def is_solvable(state):
    """Check if the puzzle is solvable."""
    state = state.flatten()
    inversions = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] > state[j] and state[i] != 0 and state[j] != 0:
                inversions += 1
    return inversions % 2 == 0

def find_blank(state):
    """Find the position of the blank (0) in the puzzle."""
    return np.argwhere(state == 0)[0]

def move_blank(state, direction):
    """Move the blank in the specified direction."""
    new_state = state.copy()
    blank_pos = find_blank(state)
    x, y = blank_pos
    if direction == "up" and x > 0:
        new_state[x, y], new_state[x - 1, y] = new_state[x - 1, y], new_state[x, y]
    elif direction == "down" and x < 2:
        new_state[x, y], new_state[x + 1, y] = new_state[x + 1, y], new_state[x, y]
    elif direction == "left" and y > 0:
        new_state[x, y], new_state[x, y - 1] = new_state[x, y - 1], new_state[x, y]
    elif direction == "right" and y < 2:
        new_state[x, y], new_state[x, y + 1] = new_state[x, y + 1], new_state[x, y]
    return new_state

def get_neighbors(state):
    """Generate all possible moves from the current state."""
    directions = ["up", "down", "left", "right"]
    neighbors = []
    for direction in directions:
        new_state = move_blank(state, direction)
        if not np.array_equal(new_state, state):
            neighbors.append(new_state)
    return neighbors

def misplaced_tiles(state, goal):
    """Calculate the Misplaced Tiles heuristic."""
    return np.sum(state != goal) - 1  # Exclude the blank tile

def manhattan_distance(state, goal):
    """Calculate the Manhattan Distance heuristic."""
    distance = 0
    for x in range(3):
        for y in range(3):
            value = state[x, y]
            if value != 0:
                goal_x, goal_y = np.argwhere(goal == value)[0]
                distance += abs(x - goal_x) + abs(y - goal_y)
    return distance

# Search algorithms
def dfs(start, goal):
    """Depth-First Search."""
    stack = [(start, [])]
    visited = set()
    while stack:
        state, path = stack.pop()
        if np.array_equal(state, goal):
            return path
        visited.add(state.tobytes())
        for neighbor in get_neighbors(state):
            if neighbor.tobytes() not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None

def bfs(start, goal):
    """Breadth-First Search."""
    queue = Queue()
    queue.put((start, []))
    visited = set()
    while not queue.empty():
        state, path = queue.get()
        if np.array_equal(state, goal):
            return path
        visited.add(state.tobytes())
        for neighbor in get_neighbors(state):
            if neighbor.tobytes() not in visited:
                queue.put((neighbor, path + [neighbor]))
    return None

def a_star(start, goal, heuristic):
    """A* Search."""
    pq = PriorityQueue()
    pq.put((0, start.tolist(), []))  # Convert NumPy array to list for comparison
    visited = set()
    while not pq.empty():
        _, state_list, path = pq.get()
        state = np.array(state_list)  # Convert back to NumPy array
        if np.array_equal(state, goal):
            return path
        visited.add(state.tobytes())
        for neighbor in get_neighbors(state):
            if neighbor.tobytes() not in visited:
                cost = len(path) + 1
                h = heuristic(neighbor, goal)
                pq.put((cost + h, neighbor.tolist(), path + [neighbor.tolist()]))
    return None

# Main function
if __name__ == "__main__":
    # Define the start and goal states
    start = np.array([[1, 2, 3],
                      [4, 0, 5],
                      [7, 8, 6]])
    goal = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 0]])

    if not is_solvable(start):
        print("The puzzle is not solvable.")
    else:
        print("Solving with DFS...")
        dfs_path = dfs(start, goal)
        print("DFS Solution:", dfs_path)

        print("\nSolving with BFS...")
        bfs_path = bfs(start, goal)
        print("BFS Solution:", bfs_path)

        print("\nSolving with A* (Misplaced Tiles)...")
        astar_mt_path = a_star(start, goal, misplaced_tiles)
        print("A* (Misplaced Tiles) Solution:", astar_mt_path)

        print("\nSolving with A* (Manhattan Distance)...")
        astar_md_path = a_star(start, goal, manhattan_distance)
        print("A* (Manhattan Distance) Solution:", astar_md_path)