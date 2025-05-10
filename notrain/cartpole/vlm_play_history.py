import ollama
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque # For efficient history management

ollama.pull('gemma3:4b')

env = gym.make('CartPole-v1', render_mode="rgb_array")
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
ax.set_title(f"CartPole-v1 | Step: {step} | Reward: 0") # Initial title
plt.axis('off') # Hide axes for cleaner look

plt.ion()  # Turn on interactive mode
plt.show()

while not terminated and not truncated and step < 200:
    step += 1
    print(f"\n--- Step {step} ---")

    # Render and update plot
    frame = env.render()
    img_plot.set_data(frame)
    ax.set_title(f"CartPole-v1 | Step: {step} | Current Reward: {reward if 'reward' in locals() else 0} | Total Steps: {step}")
    plt.draw()
    plt.pause(0.001)

    filename = f'cartpole_step_{step}.png'
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
Current CartPole Environment State:
  Position: {observation[0]:.4f} (from -2.4 to 2.4)
  Velocity: {observation[1]:.4f} (negative is left, positive is right)
  Pole Angle: {observation[2]:.4f} radians (0 is upright, +/- 0.209 rad terminates)
  Pole Angular Velocity: {observation[3]:.4f} radians/sec (negative is left, positive is right)

{history_str}

Based on the current state and past history, what action should the cart take?
Respond with only '0' for moving the cart left, or '1' for moving the cart right.
Do not include any other text, explanation, or punctuation.
"""

    action = 0 # Default action in case LLM fails or returns invalid
    try:
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': prompt_content}],
        )
        action_text = response['message']['content'].strip()

        # Try to extract the action from the LLM's response
        if "0" in action_text:
            action = 0
        elif "1" in action_text:
            action = 1
        else:
            print(f"LLM returned an invalid action: '{action_text}'. Defaulting to action 0.")
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

    except ollama.OllamaAPIError as e:
        print(f"Error communicating with Ollama: {e}")
        terminated = True  # End the episode if there's an error
    except Exception as e: # Catch other potential errors during parsing/stepping
        print(f"An unexpected error occurred: {e}")
        terminated = True

env.close()
plt.ioff() # Turn off interactive mode
plt.close(fig) # Close the plot window gracefully
print("\n--- Episode Finished ---")