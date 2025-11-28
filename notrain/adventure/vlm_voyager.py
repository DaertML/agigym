import ollama
import gymnasium as gym
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from collections import deque
import os
import ast
import heapq

# ----------------- Game State & Tooling Setup -----------------
class MockGameState:
    """A mock representation of the game world for tool functions."""
    def __init__(self, player_pos=(125, 80)):
        self.player_pos = player_pos
        self.grid = np.zeros((250, 160)) # A simple grid
        self.objects = {
            "gold key": (100, 50),
            "chalice": (150, 100),
            "red castle": (125, 80),
            "black castle": (10, 10),
            "gray castle": (200, 150)
        }
        self.player_has_object = False

    def get_object_pos(self, obj_name):
        return self.objects.get(obj_name)
    
    def move_player_to(self, new_pos):
        self.player_pos = new_pos
        print(f"Player moved to new position: {self.player_pos}")
    
    def get_player_pos(self):
        return self.player_pos

# We'll use this mock state in our tools
game_state = MockGameState()
terminated = False
truncated = False
step =1
def a_star_pathfinding(start_pos, target_pos, grid):
    """
    A* pathfinding algorithm.
    Args:
        start_pos (tuple): Starting (y, x) coordinates.
        target_pos (tuple): Target (y, x) coordinates.
        grid (np.array): A grid representing the game map.
    Returns:
        list: A list of atomic actions (integers) representing the path.
    """
    y_max, x_max = grid.shape
    open_set = [(0, start_pos)]  # (f_score, position)
    came_from = {}
    g_score = {start_pos: 0}
    f_score = {start_pos: np.linalg.norm(np.array(start_pos) - np.array(target_pos))}

    while open_set:
        current_f, current_pos = heapq.heappop(open_set)

        if current_pos == target_pos:
            path = []
            while current_pos in came_from:
                prev_pos = came_from[current_pos]
                dy = current_pos[0] - prev_pos[0]
                dx = current_pos[1] - prev_pos[1]
                # Map dx, dy to atomic actions
                if dy == -1 and dx == 0: path.append(2) # UP
                elif dy == 1 and dx == 0: path.append(5) # DOWN
                elif dy == 0 and dx == 1: path.append(3) # RIGHT
                elif dy == 0 and dx == -1: path.append(4) # LEFT
                current_pos = prev_pos
            return path[::-1] # Reverse path to get correct order

        # Consider neighbors
        for dy, dx, action in [(-1, 0, 2), (1, 0, 5), (0, 1, 3), (0, -1, 4)]:
            neighbor_pos = (current_pos[0] + dy, current_pos[1] + dx)
            if not (0 <= neighbor_pos[0] < y_max and 0 <= neighbor_pos[1] < x_max):
                continue

            tentative_g_score = g_score[current_pos] + 1
            if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                came_from[neighbor_pos] = current_pos
                g_score[neighbor_pos] = tentative_g_score
                h_score = np.linalg.norm(np.array(neighbor_pos) - np.array(target_pos))
                f_score[neighbor_pos] = tentative_g_score + h_score
                heapq.heappush(open_set, (f_score[neighbor_pos], neighbor_pos))
    return []

def navigate_to(target_pos, game_state):
    """
    Navigates the agent to a target position using A*.
    Args:
        target_pos (tuple): The (y, x) coordinates to navigate to.
        game_state: A mock representation of the game state.
    """
    print(f"Navigating to {target_pos}...")
    path_actions = a_star_pathfinding(game_state.get_player_pos(), target_pos, game_state.grid)
    
    if not path_actions:
        print("Could not find a path.")
        return "Navigation failed."

    global observation, reward, terminated, truncated, info, total_reward, step
    for action in path_actions:
        observation, reward, terminated, truncated, info = env.step(action)
        game_state.move_player_to( (game_state.get_player_pos()[0] + (action == 5) - (action == 2),
                                    game_state.get_player_pos()[1] + (action == 3) - (action == 4)) )
        total_reward += reward
        step += 1
        print(f"Executing step action: {action}, new reward: {reward}")
        time.sleep(0.05)
        if terminated or truncated:
            break
    
    return f"Navigation to {target_pos} complete."

def get_object(object_name, game_state):
    """
    Finds and picks up a specified object.
    Args:
        object_name (str): The name of the object to pick up.
        game_state: A mock representation of the game state.
    """
    print(f"Attempting to get the {object_name}...")
    obj_pos = game_state.get_object_pos(object_name)
    if not obj_pos:
        return f"Object '{object_name}' not found in game state."
    
    navigate_to(obj_pos, game_state)
    
    print("Executing FIRE action to pick up object...")
    global observation, reward, terminated, truncated, info, total_reward, step
    observation, reward, terminated, truncated, info = env.step(1) # FIRE
    total_reward += reward
    step += 1
    game_state.player_has_object = True
    
    return f"Successfully acquired the {object_name}."

def use_key(key_name, lock_location_name, game_state):
    """
    Navigates to a lock and uses a key.
    Args:
        key_name (str): The name of the key to use.
        lock_location_name (str): The name of the lock location.
        game_state: A mock representation of the game state.
    """
    print(f"Navigating to {lock_location_name} to use {key_name}...")
    lock_pos = game_state.get_object_pos(lock_location_name)
    if not lock_pos:
        return f"Lock location '{lock_location_name}' not found."
    
    navigate_to(lock_pos, game_state)
    
    print("Executing FIRE action to use the key...")
    global observation, reward, terminated, truncated, info, total_reward, step
    observation, reward, terminated, truncated, info = env.step(1) # FIRE
    total_reward += reward
    step += 1
    
    return f"Key '{key_name}' used successfully at {lock_location_name}."

# Initial set of available tools
available_tools = {
    'navigate_to': navigate_to,
    'get_object': get_object,
    'use_key': use_key
}

# ----------------- LLM Interaction and Tool Execution -----------------
def extract_code_and_tool(model_output):
    """Extracts Python code and a single tool call from the model output."""
    code_block_match = re.search(r"```python\n(.*?)```", model_output, re.DOTALL)
    tool_call_match = re.search(r"\\boxed{(.*?)}", model_output, re.DOTALL)

    code_str = code_block_match.group(1) if code_block_match else None
    tool_call_str = tool_call_match.group(1).strip() if tool_call_match else None
    
    return code_str, tool_call_str

def execute_tool(tool_call_str, available_tools, game_state):
    """Dynamically executes a tool function based on a string."""
    try:
        # Safely evaluate the function call from the string
        node = ast.parse(tool_call_str, mode='eval')
        func_call = node.body
        func_name = func_call.func.id
        
        args = [ast.literal_eval(arg) for arg in func_call.args]
        
        if func_name in available_tools:
            tool_function = available_tools[func_name]
            result = tool_function(*args, game_state)
            print(f"Tool '{func_name}' executed successfully. Result: {result}")
            return result
        else:
            print(f"Error: Unknown tool '{func_name}'.")
            return f"Error: Unknown tool '{func_name}'."
    except Exception as e:
        print(f"Failed to execute tool '{tool_call_str}'. Error: {e}")
        return f"Failed to execute tool '{tool_call_str}'. Error: {e}"

# ----------------- Main Loop -----------------
while not terminated and not truncated and step < 500:
    step += 1
    print(f"\n--- Step {step} ---")

    frame = env.render()
    img_plot.set_data(frame)
    ax.set_title(f"ALE/Adventure-v5 | Step: {step} | Current Reward: {reward if 'reward' in locals() else 0} | Total Reward: {total_reward}")
    plt.draw()
    plt.pause(0.001)

    filename = f'adventure_step_{step}.png'
    plt.savefig(filename)
    print(f"Rendered frame saved as {filename}")
    
    history_str = ""
    if history_buffer:
        history_str += "Past actions and observations (oldest to newest):\n"
        for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
            history_str += f"  Step {step - len(history_buffer) + i}: Obs={np.round(obs_hist.flatten()[:5], 2)}..., Action={action_hist}, Reward={reward_hist}\n"
    
    tools_list = "\n".join([f"- {name}: {func.__doc__.splitlines()[1].strip()}" for name, func in available_tools.items()])
    
    prompt_content = f"""
You are an intelligent agent designed to play the Adventure game. Your goal is to find the chalice and return it to the golden castle.

Current Game State:
  Player position: {game_state.get_player_pos()}
  Player has object: {game_state.player_has_object}
  Observation (partial): {np.round(observation.flatten()[:20], 2)}... (shape: {observation.shape})
  {history_str}

Available Tools (Python functions you can call):
{tools_list}

You must respond in one of two ways:
1. If you need to create a new tool, write the Python code in a code block and then in a separate boxed LaTeX block, call the tool you want to execute (either new or existing).
2. If an existing tool is sufficient, simply call the tool in a boxed LaTeX block.

Example 1 (Creating a new tool and calling it):
```python
def find_and_get_chalice(game_state):
    \"\"\"Navigates to the chalice and picks it up.\"\"\"
    navigate_to((150, 100), game_state)
    get_object("chalice", game_state)

Then, on a new line: boxedfind_and_get_chalice(game_state)

Example 2 (Calling an existing tool):
boxednavigate_to((100,50),game_state)

Based on the game state and the attached image, what is your next high-level action?
"""

try:
    response = ollama.chat(
        model='gemma3:4b',
        messages=[{'role': 'user', 'content': prompt_content, 'images': [filename]}]
    )
    model_output = response['message']['content'].strip()
    print(f"LLM's raw output:\n{model_output}")

    code_str, tool_call_str = extract_code_and_tool(model_output)

    if code_str:
        try:
            exec(code_str, globals(), available_tools)
            print("New tool generated and added to available tools.")
        except Exception as e:
            print(f"Failed to compile new tool code. Error: {e}")

    if tool_call_str:
        execute_tool(tool_call_str, available_tools, game_state)
    else:
        print("No tool call found in LLM output. Defaulting to NOOP.")
        observation, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        time.sleep(0.05)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    terminated = True
finally:
    if os.path.exists(filename):
        os.remove(filename)

env.close()
plt.ioff()
plt.close(fig)
print("\n--- Episode Finished ---")
print(f"Total Reward: {total_reward}")