import ollama
import gym
import matplotlib.pyplot as plt
import numpy as np
import time  # To introduce a small delay for visualization

ollama.pull('granite3.2-vision:2b')

env = gym.make('CartPole-v1', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0

while not terminated and not truncated and step < 200:  # Let's run for a maximum of 200 steps
    step += 1
    print(f"\n--- Step {step} ---")
    plt.imshow(env.render())
    filename = f'cartpole_step_{step}.png'
    plt.savefig(filename)
    print(f"Rendered frame saved as {filename}")

    # Prepare the prompt for the LLM with the current observation (as text)
    prompt = f"The current state of the CartPole environment is: {observation}. What action should be taken? (0 for left, 1 for right)"

    try:
        response = ollama.chat(
            model='granite3.2-vision:2b',
            messages=[{'role': 'user', 'content': prompt}],
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

        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        time.sleep(0.1) # Small delay to visualize the steps if you run it without saving

    except ollama.OllamaAPIError as e:
        print(f"Error communicating with Ollama: {e}")
        terminated = True # End the episode if there's an error

env.close()
print("\n--- Episode Finished ---")