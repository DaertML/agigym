import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import ollama
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from collections import deque
from io import BytesIO
import os

# Ensure the model is pulled
ollama.pull('gemma3:4b')

# Setup the Super Mario Bros environment without render_mode
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Correct the reset call to expect only one return value
observation = env.reset()
terminated = False
truncated = False
step = 0
total_reward = 0

HISTORY_LENGTH = 3
history_buffer = deque(maxlen=HISTORY_LENGTH)

fig, ax = plt.subplots(figsize=(10, 8))
img_plot = ax.imshow(env.render(mode="rgb_array"))
ax.set_title(f"Super Mario Bros | Step: {step} | Reward: {total_reward}")
plt.axis('off')

plt.ion()
plt.show()

def extract_integers(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    integer_strings = re.findall(r'\d+', text)
    integers = "".join(s for s in integer_strings)
    return int(integers)

def extract_action_from_latex(model_output):
    match = re.search(r"\\boxed\{(\d+)\}", model_output)
    if match:
        action_value = int(match.group(1))
        return action_value
    else:
        return extract_integers(model_output)

while not terminated and not truncated and step < 500:
    step += 1
    print(f"\n--- Step {step} ---")

    try:
        frame = env.render(mode="rgb_array")
        img_plot.set_data(frame)
        ax.set_title(f"Super Mario Bros | Step: {step} | Current Reward: {reward if 'reward' in locals() else 0} | Total Reward: {total_reward}")
        plt.draw()
        plt.pause(0.001)

        filename = f'mario_step_{step}.png'
        plt.savefig(filename)
        print(f"Rendered frame saved as {filename}")

        history_str = ""
        if history_buffer:
            history_str += "Past actions and observations (oldest to newest):\n"
            for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
                history_str += f"  Step {step - len(history_buffer) + i}: Obs={np.round(obs_hist.flatten()[:5], 2)}..., Action={action_hist}, Reward={reward_hist}\n"

        prompt_content = f"""
Current Super Mario Bros Environment State:
  Observation (partial): {np.round(observation.flatten()[:20], 2)}... (shape: {observation.shape})

{history_str}

Based on the current game state, past history, and the visual information in the attached image, what action should Mario take?
Consider the goal of moving right to complete the level, jumping to avoid obstacles and enemies, and collecting coins.

The possible reduced actions are:
0: NOOP (no operation)
1: right
2: right + A (jump)
3: right + B (run)
4: right + A + B (run and jump)
5: A (jump)
6: left
7: left + A (jump)
8: left + B (run)
9: left + A + B (run and jump)

Before making a decision analyze the given image with the capture of the game,
and evaluate the state of the environment; the image capture will let you know many details of the environment,
such as the position of Mario, the enemies, the coins, and obstacles.

Try to reason about the best action to take to navigate the level, avoid enemies, and collect coins.
Write the chosen action in boxed LaTeX format, for example: $\boxed{{1}}$
"""

        action = 0
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': prompt_content, 'images': [filename]}]
        )
        action_text = response['message']['content'].strip()

        extracted_action = extract_action_from_latex(action_text)
        if extracted_action is not None and 0 <= extracted_action <= 9:
            print(f"The extracted action is: {extracted_action}")
            action = extracted_action
        else:
            print(f"Could not extract a valid action from the model output: '{action_text}'. Defaulting to action 0.")
            action = 0

        print(f"LLM suggested action: {action} (interpreted from '{action_text}')")

        prev_observation = observation
        # Correctly unpack the 4 values returned by the old env.step()
        observation, reward, done, info = env.step(action)
        terminated = done
        truncated = False
        total_reward += reward
        print(f"Observation shape: {observation.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        history_buffer.append((prev_observation, action, reward))

        time.sleep(0.05)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Terminate the loop on error
        terminated = True

env.close()
plt.ioff()
plt.close(fig)
print("\n--- Episode Finished ---")
print(f"Total Reward: {total_reward}")
