import gymnasium as gym
import time
import os
import pygame  # Import pygame for interactive input
import tkinter as tk  # Import tkinter for GUI input
from tkinter import simpledialog

def play_and_record_cliffwalking(save_path="cliffwalking_screenshots"):
    """
    Plays the CliffWalking-v0 environment based on *interactive* user input,
    renders each step, and saves screenshots.  Uses a GUI for input.

    Args:
        save_path (str, optional): Path to save the screenshots.
            Defaults to "cliffwalking_screenshots".
    """
    # Create the environment
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")  # Use rgb_array for rendering

    # Reset the environment
    observation, info = env.reset()

    # Create the directory to save screenshots if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize pygame for event handling (for display)
    pygame.init()
    screen = pygame.display.set_mode((400, 300))  # Create a dummy screen for display
    pygame.display.set_caption("CliffWalking Interactive")
    clock = pygame.time.Clock() # Add a clock

    # Initialize tkinter for GUI input
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    i = 0  # Step counter
    done = False  # Episode termination flag

    # Game loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                return

        # Get action from a dialog box
        action = simpledialog.askinteger("CliffWalking",
                                         f"Enter action (0: Up, 1: Right, 2: Down, 3: Left):\n Current Observation: {observation}",
                                         minvalue=0, maxvalue=3)

        if action is None:  # Check if the user cancelled the dialog
            print("User cancelled input, exiting.")
            pygame.quit()
            env.close()
            return

        # Render the environment and save the frame
        frame = env.render()
        if frame is not None:
            # Display the frame using pygame
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1)) # Important
            screen.blit(surface, (0, 0))
            pygame.display.flip() # Update
            clock.tick(30) # Add a tick

            frame_filename = f"{save_path}/step_{i}.png"
            import imageio
            imageio.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
        else:
            print(f"Frame is None at step {i}, skipping save")

        # Take the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}, Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        i += 1  # Increment step counter

        # Check if the episode is done
        if terminated or truncated:
            print("Episode finished.")
            done = True

    # Close the environment and pygame
    env.close()
    pygame.quit()



if __name__ == "__main__":
    # Play the environment and record screenshots
    play_and_record_cliffwalking()
    print("Done!")