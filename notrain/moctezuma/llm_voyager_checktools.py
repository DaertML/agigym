import ollama
import gymnasium as gym
import numpy as np
import time
import re
import json
import cv2
from collections import deque
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Union, List
import ale_py

# ----------------- Pydantic Data Models for Structured LLM Output -----------------
# Data model for using an existing tool
class UseToolPlan(BaseModel):
    decision: Literal["use_tool"]
    tool_name: str
    action_id: int = Field(None, description="The specific action ID for the act tool.")
    reasoning: str = Field(min_length=1)

# Data model for creating a new tool with a list of actions
class CreateToolWithActionsPlan(BaseModel):
    decision: Literal["create_tool"]
    tool_name: str = Field(min_length=1)
    tool_description: str = Field(min_length=1)
    actions: List[int] = Field(..., description="A list of integer action IDs to be combined into the new tool.")
    reasoning: str = Field(min_length=1)

# The Union type is for type hinting, the validation will be done manually
Plan = Union[UseToolPlan, CreateToolWithActionsPlan]

# ----------------- Utility Functions -----------------
def extract_json_from_response(model_output: str) -> dict:
    """
    Extracts a JSON object from a string that may contain extra text.
    It looks for a fenced code block first, then falls back to the whole string.
    """
    # Try to find a JSON code block first
    json_match = re.search(r"```json\n(.*?)\n```", model_output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # If no code block, assume the entire output is a JSON string
        json_str = model_output.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return {}

# ----------------- Core Environment Setup -----------------
ollama.pull('gemma3:4b')

env = gym.make('ALE/MontezumaRevenge-v5', render_mode="rgb_array")
observation, info = env.reset()
terminated = False
truncated = False
step = 0
total_reward = 0
HISTORY_LENGTH = 5
history_buffer = deque(maxlen=HISTORY_LENGTH)

# ----------------- Skill/Tool Library -----------------
class ToolLibrary:
    def __init__(self, env):
        self.env = env
        self.action_mapping = {
            0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
            6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT", 10: "UPFIRE",
            11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE", 14: "UPRIGHTFIRE",
            15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
        }

    def act(self, action_id: int):
        """Performs a specific action in the environment."""
        action_name = self.action_mapping.get(action_id, "UNKNOWN")
        print(f"Executing action: {action_name} (ID: {action_id})")
        return self.env.step(action_id)

    def get_state_description(self, total_reward):
        """
        Returns a textual description of the current game state.
        This is a placeholder for a real vision model.
        """
        # Get player position from RAM
        ale = self.env.unwrapped.ale
        player_x = ale.getRAM()[15]
        player_y = ale.getRAM()[14]

        # Simple logic based on player position
        if total_reward > 0:
            return f"The agent has a score of {total_reward}. It has likely acquired a key and is in a new room. Player position: ({player_x}, {player_y})."
        
        if player_x > 70 and player_y > 100:
            return f"The agent is on the bottom platform in the main chamber. A ladder is to the right. Player position: ({player_x}, {player_y})."
        elif player_x < 70 and player_y < 100:
            return f"The agent is on the top platform. A ladder is below and a rope is to the right. Player position: ({player_x}, {player_y})."
        else:
            return f"The agent is in the main chamber, near the ladder. Player position: ({player_x}, {player_y})."

# Initialize the tool library
tool_library = ToolLibrary(env)

# ----------------- Planning and Code Generation Agents -----------------
def generate_plan(model, state_description, history, sub_goal, available_tools, failure_message=""):
    """
    The Planner Agent: Decides on the next action or if a new tool is needed.
    It returns a validated plan dictionary.
    """
    tools_str = "\n".join([f"  - {tool}" for tool in available_tools])
    
    use_tool_schema = UseToolPlan.model_json_schema()
    create_tool_schema = CreateToolWithActionsPlan.model_json_schema()

    prompt = f"""
    You are a planning agent for a reinforcement learning task in the Montezuma's Revenge game.
    Your goal is to guide the agent to acquire the treasure. This is a hard exploration game.
    You should prefer creating new tools that combine multiple actions to solve complex sub-problems,
    like climbing a ladder, jumping a gap, or acquiring a key. This will make the agent more efficient.

    Current High-Level Task: {sub_goal}

    Current Environment State:
    {state_description}

    Recent History:
    {history}

    Available Tools:
    {tools_str}
    
    The 'act' tool requires a single integer parameter from 0 to 17.

    {failure_message}

    Based on the state and history, decide on the next step.
    Your response MUST be a JSON object that adheres to one of the following two schemas.
    
    Schema 1: Use an existing tool. This is for simple, atomic actions.
    ```json
    {json.dumps(use_tool_schema, indent=2)}
    ```

    Schema 2: Create a new tool. This is for combining actions into a more complex behavior.
    ```json
    {json.dumps(create_tool_schema, indent=2)}
    ```

    Your response should contain ONLY the JSON object, enclosed in a fenced code block like this:
    ```json
    {{...}}
    ```
    """
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.7}
    )
    
    response_text = response['message']['content']
    raw_json = extract_json_from_response(response_text)
    
    try:
        # Manually attempt to validate against each model
        if "decision" in raw_json and raw_json["decision"] == "use_tool":
            validated_plan = UseToolPlan.model_validate(raw_json)
        elif "decision" in raw_json and raw_json["decision"] == "create_tool":
            validated_plan = CreateToolWithActionsPlan.model_validate(raw_json)
        else:
            raise ValidationError("Invalid 'decision' field in JSON.")
            
        return validated_plan.model_dump()
        
    except ValidationError as e:
        print("Planner response failed Pydantic validation. Retrying...")
        print(f"Validation Error: {e}")
        return {"decision": "use_tool", "tool_name": "act", "action_id": 3, "reasoning": "Fallback to a safe 'RIGHT' action due to invalid response."}

def create_new_tool(tool_name: str, actions: List[int], tool_library: ToolLibrary):
    """
    Generates and executes the code for a new tool based on a list of actions.
    """
    # Construct the Python code for the new tool method with proper indentation
    action_calls = "\n".join([f"        self.act({action})" for action in actions])
    tool_code = f"""
def {tool_name}(self):
        \"\"\"Automatically generated tool with a sequence of actions.\"\"\"
{action_calls}
"""
    
    print("\n--- Generated Tool Code ---")
    print(tool_code)
    print("---------------------------\n")

    # Safely execute the generated tool code and add it to the ToolLibrary
    try:
        local_scope = {"self": tool_library, "env": tool_library.env}
        exec(tool_code, {}, local_scope)
        setattr(ToolLibrary, tool_name, local_scope[tool_name])
        print(f"Successfully created and added new tool: {tool_name}")
        return True
    except Exception as e:
        print(f"Error executing generated code for tool '{tool_name}': {e}")
        return False

# ----------------- Main Loop -----------------
high_level_tasks = [
    "Navigate to the main chamber and climb down the ladder.",
    "Jump over the skull.",
    "Acquire the key.",
    "Open the door and move to the next room."
]
current_task = high_level_tasks[0]
available_tools = [
    'act', 'get_state_description'
]

llm_model = 'gemma3:4b'
failure_message = ""

# Create a window to display the game
cv2.namedWindow('Montezuma\'s Revenge')

while not terminated and not truncated and step < 200:
    step += 1
    print(f"\n--- Step {step} ---")
    print(f"Available tools: {available_tools}")

    # Get the current state description for the LLM
    state_description = tool_library.get_state_description(total_reward)
    print(f"Current State: {state_description}")

    # Render the environment and display it in real-time
    rgb_array = env.render()
    cv2.imshow('Montezuma\'s Revenge', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27: # 'q' or 'ESC'
        print("Exiting...")
        break
    
    history_str = ""
    if history_buffer:
        history_str += "Past actions and rewards (oldest to newest):\n"
        for i, (obs_hist, action_hist, reward_hist) in enumerate(history_buffer):
            history_str += f"  Step {step - len(history_buffer) + i}: Action={action_hist}, Reward={reward_hist}\n"
    
    plan = generate_plan(llm_model, state_description, history_str, current_task, available_tools, failure_message)
    failure_message = "" # Reset the failure message for the next turn

    if plan["decision"] == "create_tool":
        tool_name = plan["tool_name"]
        actions = plan["actions"]
        print(f"Planner suggests creating a new tool: '{tool_name}' with actions: {actions}")
        if create_new_tool(tool_name, actions, tool_library):
            available_tools.append(tool_name)
            print(f"New tool '{tool_name}' is now available.")
        else:
            failure_message = f"Your attempt to create the tool '{tool_name}' failed. The code generated had a bug. Please try to fix the code or use a different approach. The error was: {e}."
            print(failure_message)
        continue
    else: # "use_tool"
        action_to_take = plan["tool_name"]
        print(f"Planner suggests using existing tool: {action_to_take}")
        
        # Check if the tool actually exists
        if action_to_take not in available_tools:
            failure_message = f"You attempted to use tool '{action_to_take}', but it has not been created yet. You must first create it using the 'create_tool' decision."
            print(failure_message)
            continue
            
    try:
        tool_function = getattr(tool_library, action_to_take)
        
        # Pass action_id only if the tool is the base 'act' function
        if action_to_take == 'act':
            action_id = plan.get('action_id')
            if action_id is not None:
                result = tool_function(action_id)
            else:
                raise ValueError("Action ID not provided for 'act' tool.")
        else:
            # For all other tools, call with no arguments
            result = tool_function()
        
        if isinstance(result, tuple) and len(result) >= 5:
            observation, reward, terminated, truncated, info = result
            total_reward += reward
            print(f"Observation Shape: {observation.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
            history_buffer.append((observation, action_to_take, reward))
        else:
            print(f"Tool '{action_to_take}' executed.")
            pass
            
    except Exception as e:
        failure_message = f"Error executing chosen tool '{action_to_take}': {e}"
        print(failure_message)
        pass

    time.sleep(0.1)

env.close()
cv2.destroyAllWindows()
print("\n--- Episode Finished ---")