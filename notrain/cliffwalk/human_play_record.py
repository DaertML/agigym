import gymnasium as gym
import time
import os

def play_and_record_cliffwalking(actions, save_path="cliffwalking_screenshots"):
    """
    Plays the CliffWalking-v0 environment based on provided actions,
    renders each step, and saves screenshots.

    Args:
        actions (list of int): A list of actions to perform in the environment.
            Each action should be an integer:
            0: Move up
            1: Move right
            2: Move down
            3: Move left
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

    # Iterate through the provided actions
    for i, action in enumerate(actions):
        # Render the environment and save the frame
        frame = env.render()
        if frame is not None: # check if the frame is valid
            frame_filename = f"{save_path}/step_{i}.png"
            # Use a more robust way to save the image using imageio
            import imageio
            imageio.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
        else:
            print(f"Frame is None at step {i}, skipping save")

        # Take the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}, Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        # Check if the episode is done
        if terminated or truncated:
            print("Episode finished.")
            break

    # Close the environment
    env.close()



if __name__ == "__main__":
    # Define the sequence of actions to take.  For testing.
    # Example: Move right twice, down once, right 8 times.
    actions = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]

    # Play the environment and record screenshots
    play_and_record_cliffwalking(actions)
    print("Done!")