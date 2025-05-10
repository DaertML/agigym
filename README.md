# agigym
Test and train your LLMs, VLMs and other AIs in the AGI Gym

# Introduction
## Running Inference (notrain)
Choose the script that you want to try from the folder; they are organized by the environment they solve. If it says "VLM" it means, that the render of the environment is sent to the VLM, if it says human play... then you know what this is about. Just python3 <script>; make sure to have Ollama running locally and check the temp of your GPU... these things get so hot so fast.

# Disclaimer
This is WIP, in the last minutes of testing, I've realized a couple of mistakes:
1. I was not given the image to the VLM, which make the results even more impressive.
2. I was using gemma4:3b, instead of mistral-small3.1 (which I also have tested in other cases); making things even more promising :)
