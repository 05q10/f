from collections import deque 
import time 
 
# graph structure (unweighted) 
city_map = { 
   'A': ['B', 'C'], 
   'B': ['D', 'E'], 
   'C': ['F'], 
   'D': [], 
   'E': [], 
   'F': ['G'], 
   'G': [] 
} 
 
print("Graph Structure:") 
for node, neighbors in city_map.items(): 
   print(f"{node} -> {', '.join(neighbors) if neighbors else 'No outgoing edges'}") 
print() 
 
start_node = input("Enter the START node ").strip().upper() 
goal_node = input("Enter the GOAL node ").strip().upper() 
 
if start_node not in city_map or goal_node not in city_map: 
   print("Invalid start or goal node") 
   exit() 
 
def bfs(graph, start, goal): 
   visited = set() 
   queue = deque([(start, [start])]) 
   start_time = time.time() 
 
   while queue: 
       current, path = queue.popleft() 
       print(f"[BFS] Visiting: {current} | Path so far: {' -> '.join(path)}") 
       if current == goal: 
           end_time = time.time() 
           print("\nBFS Path from", start, "to", goal, ":", " -> ".join(path)) 
           print("Time Taken by BFS: {:.3f} ms".format((end_time - 
start_time) * 1000)) 
           print("Number of Nodes Visited in BFS:", len(visited) + 
1) 
           return path 
       if current not in visited: 
           visited.add(current) 
           for neighbor in graph[current]: 
               if neighbor not in visited: 
                   queue.append((neighbor, path + [neighbor])) 
   print("No BFS path found.") 
   return [] 
 
def dfs(graph, start, goal): 
   visited = set() 
   stack = [(start, [start])] 
   start_time = time.time() 
 
   while stack: 
       current, path = stack.pop() 
       print(f"[DFS] Visiting: {current} | Path so far: {' -> '.join(path)}") 
       if current == goal: 
           end_time = time.time() 
           print("\nDFS Path from", start, "to", goal, ":", " -> ".join(path)) 
           print("Time Taken by DFS: {:.3f} ms".format((end_time - 
start_time) * 1000)) 
           print("Number of Nodes Visited in DFS:", len(visited) + 
1) 
           return path 
       if current not in visited: 
           visited.add(current) 
           for neighbor in reversed(graph[current]): 
               if neighbor not in visited: 
                   stack.append((neighbor, path + [neighbor])) 
   print("No DFS path found.") 
   return [] 
 
# Run BFS and DFS using user input 
 
print("\nBreadth-First Search: ") 
bfs_path = bfs(city_map, start_node, goal_node) 
 
print("\nDepth-First Search: ") 
dfs_path = dfs(city_map, start_node, goal_node) 
 
# comparison 
print("\nBFS vs DFS Comparison: ") 
if bfs_path and dfs_path: 
   print("Length of BFS Path:", len(bfs_path)) 
   print("Length of DFS Path:", len(dfs_path)) 
   if len(bfs_path) < len(dfs_path): 
       print("BFS found a shorter path.") 
   elif len(dfs_path) < len(bfs_path): 
       print("DFS found a shorter path.") 
   else: 
       print("Both BFS and DFS found paths of equal length.") 
else: 
   print("One or both algorithms failed to find a path.")