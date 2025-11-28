from groq import Groq
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import base64

from collections import deque
from io import BytesIO
import os  # For file path manipulation

GROQ_API_KEY = "APIKEY_HERE"
client = Groq(api_key=GROQ_API_KEY)

env = gym.make('ALE/Skiing-v5', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0
total_reward = 0

HISTORY_LENGTH = 3
history_buffer = deque(maxlen=HISTORY_LENGTH)

fig, ax = plt.subplots(figsize=(10, 8))
img_plot = ax.imshow(env.render())
ax.set_title(f"ALE/Skiing-v5 | Step: {step} | Reward: {total_reward}")
plt.axis('off')

plt.ion()
plt.show()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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
    ax.set_title(f"ALE/Skiing-v5 | Step: {step} | Current Reward: {reward if 'reward' in locals() else 0} | Total Reward: {total_reward}")
    plt.draw()
    plt.pause(0.001)

    filename = f'skiing_step_{step}.png'
    plt.savefig(filename)
    print(f"Rendered frame saved as {filename}")

    history_str = ""
    if history_buffer:
        history_str += "Past actions and observations (oldest to newest):\n"
        for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
            history_str += f"  Step {step - len(history_buffer) + i}: Obs={np.round(obs_hist.flatten()[:5], 2)}..., Action={action_hist}, Reward={reward_hist}\n"

    prompt_content = """
Based on the current game state, past history, and the visual information in the attached image, what action should the player take?
Consider the goal of running through all gates (between the poles) in the fastest time.
You are penalized five seconds for each gate you miss.
If you hit a gate or a tree, you will lose time.
The possible reduced actions are:
0: NOOP (no operation)
1: RIGHT
2: LEFT

Before making a decision analyze the given image with the capture of the game,
and evaluate the state of the environment; the image capture will let you know many details of the environment.

Do not pay attention too much to the intermediate rewards, as they will go up if the character moves between the
purple poles.

Try to reason about the best action to take to navigate through the gates.
Write the chosen action in boxed LaTeX format, for example: $\boxed{1}$
"""

    action = 0
    try:
        base64_image = encode_image(filename)        
        response = client.chat.completions.create(
            #model='meta-llama/llama-4-maverick-17b-128e-instruct',
            model='meta-llama/llama-4-scout-17b-16e-instruct',
            messages=[
                {'role': 'user', 'content': [{"type": "text", "text": str(prompt_content)}]},
                {'role': 'user', 'content': [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ]
        )
        action_text = response.choices[0].message.content.strip()
        extracted_action = extract_action_from_latex(action_text)
        if extracted_action is not None and 0 <= extracted_action <= 2:
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

env.close()
plt.ioff()
plt.close(fig)
print("\n--- Episode Finished ---")
print(f"Total Reward: {total_reward}")