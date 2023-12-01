import os
import sys
sys.path.append(".")
import json
from copy import deepcopy

from exp.utils import load_config
from agents.mcts import run_episode

if __name__ == "__main__":
    path = 'configs/mcts/cartpole_budgets.yml'
    out_path = 'results/mcts/cartpole_budgets.json'
    out_dir = os.path.dirname(out_path)
    config = load_config(path)

    budgets = config['budgets']
    seeds = list(range(config['num_seeds']))

    results = {}
    for budget in budgets:
        if budget not in results:
            results[budget] = {}
        for seed in seeds:
            results[budget][seed] = {} 
            cfg = deepcopy(config)
            cfg['budget'] = budget
            cfg['seed'] = seed

            total_reward = run_episode(cfg)
            print(f"Budget : {budget}, Seed : {seed}, Total Reward : {total_reward}")
            results[budget][seed]['total_reward'] = total_reward
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as fp:
        json.dump(results, fp)
    print(results)

