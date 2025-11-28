import ollama
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from collections import deque # For efficient history management

ollama.pull('mistral-small3.1')

env = gym.make('ALE/Skiing-v5', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0
total_reward = 0

# --- History Management ---
HISTORY_LENGTH = 3  # Number of past steps to remember for Skiing
history_buffer = deque(maxlen=HISTORY_LENGTH)
# --- End History Management ---

# Initialize the plot outside the loop
fig, ax = plt.subplots(figsize=(10, 8)) # Increased size for Skiing
img_plot = ax.imshow(env.render())
ax.set_title(f"ALE/Skiing-v5 | Step: {step} | Reward: {total_reward}")
plt.axis('off') # Hide axes for cleaner look

plt.ion()  # Turn on interactive mode
plt.show()

def extract_action_from_latex(model_output):
    """
    Extracts the numerical action value from the model's output string,
    assuming it is in boxed LaTeX format.

    Args:
        model_output (str): The string output from the language model.

    Returns:
        int or None: The extracted action value as an integer, or None if not found.
    """
    match = re.search(r"\\boxed{(\d+)}", model_output)
    if match:
        action_value = int(match.group(1))
        return action_value
    else:
        return None

while not terminated and not truncated and step < 500: # Increased steps for Skiing
    step += 1
    print(f"\n--- Step {step} ---")

    # Render and update plot
    frame = env.render()
    img_plot.set_data(frame)
    ax.set_title(f"ALE/Skiing-v5 | Step: {step} | Current Reward: {reward if 'reward' in locals() else 0} | Total Reward: {total_reward}")
    plt.draw()
    plt.pause(0.001)

    filename = f'skiing_step_{step}.png'
    plt.savefig(filename)
    print(f"Rendered frame saved as {filename}")

    # --- Prepare History String for LLM ---
    history_str = ""
    if history_buffer:
        history_str += "Past actions and observations (oldest to newest):\n"
        for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
            history_str += f"  Step {step - len(history_buffer) + i}: Obs={np.round(obs_hist.flatten()[:5], 2)}..., Action={action_hist}, Reward={reward_hist}\n" # Show a snippet of observation
    # --- End Prepare History String ---

    # Prepare the prompt for the LLM with the current observation and history
    # Use a structured prompt to help the LLM understand the input
    prompt_content = f"""
Current Skiing Environment State:
  Observation (partial): {np.round(observation.flatten()[:20], 2)}... (shape: {observation.shape})

{history_str}

Based on the current game state and past history, what action should the player take?
Consider the goal of running through all gates (between the poles) in the fastest time.
You are penalized five seconds for each gate you miss.
If you hit a gate or a tree, you will lose time.
The possible reduced actions are:
0: NOOP (no operation)
1: RIGHT
2: LEFT

Try to reason about the best action to take to navigate through the gates.
Write the chosen action in boxed LaTeX format, for example: $\boxed{1}$
"""

    action = 0 # Default action
    try:
        response = ollama.chat(
            model='mistral-small3.1', # Or try a different model if needed
            messages=[{'role': 'user', 'content': prompt_content}],
        )
        action_text = response['message']['content'].strip()

        extracted_action = extract_action_from_latex(action_text)
        if extracted_action is not None and 0 <= extracted_action <= 2:
            print(f"The extracted action is: {extracted_action}")
            action = extracted_action
        else:
            print(f"Could not extract a valid action from the model output: '{action_text}'. Defaulting to action 0.")
            action = 0

        print(f"LLM suggested action: {action} (interpreted from '{action_text}')")

        # Take the step in the environment
        prev_observation = observation # Store current observation before updating
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Observation shape: {observation.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        # --- Update History Buffer ---
        history_buffer.append((prev_observation, action, reward))
        # --- End Update History Buffer ---

        time.sleep(0.05)  # Adjust delay as needed for Skiing
    except Exception as e: # Catch other potential errors during parsing/stepping
        print(f"An unexpected error occurred: {e}")
        terminated = True

env.close()
plt.ioff() # Turn off interactive mode
plt.close(fig) # Close the plot window gracefully
print("\n--- Episode Finished ---")
print(f"Total Reward: {total_reward}")