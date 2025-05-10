import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import pygame # For keyboard input

env = gym.make('CartPole-v1', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0

# --- History Management ---
HISTORY_LENGTH = 5
history_buffer = deque(maxlen=HISTORY_LENGTH)
# --- End History Management ---

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
img_plot = ax.imshow(env.render())
ax.set_title(f"CartPole-v1 | Step: {step} | Reward: 0")
plt.axis('off')
plt.ion()
plt.show()

# Initialize Pygame for event handling
pygame.init()
screen = pygame.display.set_mode((100, 100), pygame.NOFRAME) # Create a small, hidden window

def get_human_action():
    """Gets action from the human player using keyboard input."""
    action = None
    while action is None:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_ESCAPE:  # Add escape key to quit
                    return -1 # Signal to quit
        time.sleep(0.01)  # Reduce CPU usage
    return action

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

    # Get action from human player
    action = get_human_action()
    if action == -1:
        print("Exiting game.")
        break # Exit the game loop

    # --- Prepare History String for display ---
    history_str = ""
    if history_buffer:
        history_str += "Past actions and observations (oldest to newest):\n"
        for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
            history_str += f"  Step {step - len(history_buffer) + i}: Obs={np.round(obs_hist, 2)}, Action={action_hist}, Reward={reward_hist}\n"
    print(history_str)
    # --- End Prepare History String ---

    # Take the step in the environment
    prev_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Human chose action: {action}")
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    # --- Update History Buffer ---
    history_buffer.append((prev_observation, action, reward))
    # --- End Update History Buffer ---

    time.sleep(0.1)

env.close()
plt.ioff()
plt.close(fig)
pygame.quit() # Properly quit pygame
print("\n--- Episode Finished ---")