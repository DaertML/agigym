import ollama
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from collections import deque
from io import BytesIO
import os
import ale_py

ollama.pull('gemma3:12b')

env = gym.make('ALE/Adventure-v5', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0
total_reward = 0

HISTORY_LENGTH = 3
history_buffer = deque(maxlen=HISTORY_LENGTH)

fig, ax = plt.subplots(figsize=(10, 8))
img_plot = ax.imshow(env.render())
ax.set_title(f"ALE/Adventure-v5 | Step: {step} | Reward: {total_reward}")
plt.axis('off')

plt.ion()
plt.show()

def extract_action_from_latex(model_output):
    match = re.search(r"\\boxed{(\d+)}", model_output)
    if match:
        action_value = int(match.group(1))
        return action_value
    else:
        return None

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

    prompt_content = f"""
Current Adventure Environment State:
  Observation (partial): {np.round(observation.flatten()[:20], 2)}... (shape: {observation.shape})

{history_str}

Based on the current game state, past history, and the visual information in the attached image, what action should the player take?
The goal is to navigate the maze, find the chalice, return it to the yellow castle, and put it on the red square to win the game.
You must find and use three keys (gold, gray, black) to unlock corresponding castles.
You can carry one object at a time.
Watch out for enemies: the Red Dragon, the Green Dragon, and the Bat.
The possible actions are:
0: NOOP
1: FIRE (attack with sword)
2: UP
3: RIGHT
4: LEFT
5: DOWN
6: UP-RIGHT
7: UP-LEFT
8: DOWN-RIGHT
9: DOWN-LEFT
10: UP-FIRE
11: RIGHT-FIRE
12: LEFT-FIRE
13: DOWN-FIRE
14: UP-RIGHT-FIRE
15: UP-LEFT-FIRE
16: DOWN-RIGHT-FIRE
17: DOWN-LEFT-FIRE

Before making a decision, analyze the given image with the capture of the game and evaluate the state of the environment. The image capture will let you know many details of the environment, such as the location of the player, enemies, and objects.

Try to reason about the best action to take to progress towards the goal.
Write the chosen action in boxed LaTeX format, for example: $\boxed{1}$
"""

    action = 0
    try:
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': prompt_content, 'images': [filename]}]
        )
        action_text = response['message']['content'].strip()

        extracted_action = extract_action_from_latex(action_text)
        if extracted_action is not None and 0 <= extracted_action <= 17:
            print(f"The extracted action is: {extracted_action}")
            action = extracted_action
        else:
            print(f"Could not extract a valid action from the model output: '{action_text}'. Defaulting to action 0.")
            action = 0

        print(f"LLM suggested action: {action} (interpreted from '{action_text}')")

        prev_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Observation shape: {observation.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        history_buffer.append((prev_observation, action, reward))

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
