import ollama
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from collections import deque # For efficient history management

ollama.pull('gemma3:4b')

env = gym.make('CliffWalking-v0', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0

# --- History Management ---
HISTORY_LENGTH = 5  # Number of past steps to remember
history_buffer = deque(maxlen=HISTORY_LENGTH)
# --- End History Management ---

# Initialize the plot outside the loop
fig, ax = plt.subplots(figsize=(8, 6)) # Added figsize for better viewing
img_plot = ax.imshow(env.render())
ax.set_title(f"CliffWalking-v0 | Step: {step} | Reward: 0") # Initial title
plt.axis('off') # Hide axes for cleaner look

plt.ion()  # Turn on interactive mode
plt.show()

def extract_action_from_latex2(model_output):
    """
    Extracts the numerical action value from the model's output string,
    looking for digits within LaTeX boxed format or directly.

    Args:
        model_output (str): The string output from the language model.

    Returns:
        str: The extracted action value, or None if not found.
    """
    match = re.search(r"\\boxed{(\d+)}", model_output) or re.search(r"(\d+)", model_output) #changed regex
    if match:
        action_value = match.group(1)
        return action_value
    else:
        return None


def extract_action_from_latex(model_output):
    """
    Extracts the chosen action value from the model's output string,
    assuming it is in boxed LaTeX format.

    Args:
        model_output (str): The string output from the language model.

    Returns:
        str: The extracted action value, or None if not found.
    """
    match = re.search(r"\\boxed{(\d+)}", model_output)
    if match:
        action_value = match.group(1)
        return action_value
    else:
        return None

while not terminated and not truncated and step < 200:
    step += 1
    print(f"\n--- Step {step} ---")

    # Render and update plot
    frame = env.render()
    img_plot.set_data(frame)
    ax.set_title(f"CliffWalking-v0 | Step: {step} | Current Reward: {reward if 'reward' in locals() else 0} | Total Steps: {step}")
    plt.draw()
    plt.pause(0.001)

    filename = f'cliffwalk_step_{step}.png'
    plt.savefig(filename)
    print(f"Rendered frame saved as {filename}")

    # --- Prepare History String for LLM ---
    history_str = ""
    if history_buffer:
        history_str += "Past actions and observations (oldest to newest):\n"
        for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
            history_str += f"  Step {step - len(history_buffer) + i}: Obs={np.round(obs_hist, 2)}, Action={action_hist}, Reward={reward_hist}\n"
    # --- End Prepare History String ---

    # Prepare the prompt for the LLM with the current observation and history
    # Use a structured prompt to help the LLM understand the input
    prompt_content = f"""
Current CliffWalking-v0 Environment State:
  Position: {observation:.4f} (2D Grid)

{history_str}

Based on the current state and past history, what action should the player take?
Do not repeat the actions that produce a bad reward, try to reason about it.
Avoid getting stuck on a square, if an action did not improve your reward for
a given set of steps, try different options; avoid the cliff (in brown) and get
to the cookie (goal). If you try to access a place that can not be accessed you
will get an invalid movement, try a different move.
0: move up
1: move right
2: move down
3: move left.
Write the chosen action in boxed LaTeX format, for example: $\boxed{1}$
"""

    action = 0 # Default action in case LLM fails or returns invalid
    try:
        response = ollama.chat(
            model='mistral-small3.1',
            messages=[{'role': 'user', 'content': prompt_content}],
        )
        action_text = response['message']['content'].strip()

        extracted_action = extract_action_from_latex(action_text)
        if extracted_action:
            print(f"The extracted action is: {extracted_action}")
            action = int(extracted_action)
        else:
            print("Could not extract the chosen action from the model output.")
            action = 0

        print(f"LLM suggested action: {action} (interpreted from '{action_text}')")

        # Take the step in the environment
        prev_observation = observation # Store current observation before updating
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        # --- Update History Buffer ---
        history_buffer.append((prev_observation, action, reward))
        # --- End Update History Buffer ---

        time.sleep(0.1)  # Small delay
    except Exception as e: # Catch other potential errors during parsing/stepping
        print(f"An unexpected error occurred: {e}")
        terminated = True

env.close()
plt.ioff() # Turn off interactive mode
plt.close(fig) # Close the plot window gracefully
print("\n--- Episode Finished ---")