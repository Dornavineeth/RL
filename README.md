# RL
Welcome to the implementation repository of well known reinforcement learning algorithms! In this project, we delve into the intricacies of two key algorithms, Monte Carlo Tree Search (MCTS) and Priority Sweeping, providing comprehensive insights into their functionalities and effectiveness.

# Setup

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

# Running experiments of MCTS

```
# CartPole : Return vs Budget
bash scripts/mcts_cartpole_budgets.sh

# Grid World : Return vs Budget
bash scripts/mcts_gridworld_budgets.sh

# CartPole : Return vs Depth
bash scripts/mcts_cartpole_depth.sh

# Grid World : Return vs Depth
bash scripts/mcts_gridworld_depth.sh

```

# Running experiments of Priority Sweeping
To run the algorithms, Please set the optimal hyper parameters in the main function

```
# Grid World
python PS_grid.py

# Cliff Walking
python cliff.py
```
