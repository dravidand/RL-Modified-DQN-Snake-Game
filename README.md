# 🐍 CW2 - RL Modified DQN Snake Game

This project is an advanced implementation of the classic Snake Game using **Reinforcement Learning (RL)**, specifically a **Modified Deep Q-Network (DQN)**. Developed as part of a group coursework submission, the project showcases how combining RL with smarter **exploration strategies** results in:

* 🚀 **Improved learning efficiency**
* 🧠 **Adaptive agent behavior**
* 🎯 **Consistent high-score performance**

### 🔧 What This Code Does

* Implements the **Snake game environment** using PyGame.
* Develops a **custom DQN agent** entirely from scratch (see `DQN.py`).
* Integrates **alternative exploration strategies** beyond the standard ε-greedy — such as **Boltzmann softmax** — for better action selection during training.
* Uses **experience replay** and a **target network** to stabilize learning.
* YAML is used for parameter configuration, implemented independently.

* The agent was trained for 200 episodes, and a sample gameplay video is attached below to demonstrate the results. 🎮
  https://github.com/user-attachments/assets/b952315d-bac3-45f0-a2cc-30d36feb3213
  ![image](https://github.com/user-attachments/assets/71bc8fab-a3d3-422a-b55d-89daa80a600a)
  Trained for about 200 episodes long with score
  
  ![image](https://github.com/user-attachments/assets/37075163-164b-407e-b7d4-852be40fdf27)
  Screenshot of snake eating apple!


  

### 👥 Contributors

This project was a team effort. Major contributions to the repository were made by:

* **Dravidan**
* **Yana**

---

### 📚 References

* Snake Game Environment:
  [SumitJainUTD/pytorch-ann-snake-game-ai](https://github.com/SumitJainUTD/pytorch-ann-snake-game-ai)
  *Environment structure borrowed; agent and exploration logic reimplemented from scratch.*

* Exploration Strategy (Boltzmann Softmax):
  [Kaggle - Deep Reinforcement Learning Part 4](https://www.kaggle.com/code/yashsahu/deep-reinforcement-learning-part-4)
  *Used for conceptual reference, with implementation written independently.*

* YAML File Handling:
  [Python Land - YAML Processing](https://python.land/data-processing/python-yaml)
  *Only the ideology was referenced; YAML integration coded from scratch.*

> **Note:** Package and version requirements are listed in the `requirements.txt` file.
