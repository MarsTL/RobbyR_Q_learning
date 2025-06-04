# Robby the Robot Q-Learning Simulation

# Overview
This project implements Q-learning algorithm to simulate Robby the Robot (reinforcement learning agent) who learns to collect 
cans in 10x10 grid. He senses his surroundings and learn from rewards and penalties, and get the most optimal strategy over time. 
The grid populates with soda cans (C) and grid's surrounding by  walls (#). His sensors let him detect the content of his current 
cell and the four adjacent calls (north, south, east, west). 

# Run

1. Make sure Python 3 installed.
2. Install required packages:
   ```bash
   pip install numpy matplotlib
3. python3 train_robby.py
