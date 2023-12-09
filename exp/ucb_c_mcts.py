import os
import sys
sys.path.append(".")
import json
from copy import deepcopy

from exp.utils import load_config
from agents.mcts import run_episode

if __name__ == "__main__":
    path = 'configs/mcts/cartpole_ucb_c.yml'
    out_path = 'results/mcts/cartpole_ucb_c.json'
    out_dir = os.path.dirname(out_path)
    config = load_config(path)

    ucb_c = config['ucb_c']
    seeds = list(range(config['num_seeds']))

    results = {}
    for c in ucb_c:
        if c not in results:
            results[c] = {}
        for seed in seeds:
            results[c][seed] = {} 
            cfg = deepcopy(config)
            cfg['ucb_c'] = c
            cfg['seed'] = seed

            total_reward = run_episode(cfg)
            print(f"UCB c : {c}, Seed : {seed}, Total Reward : {total_reward}")
            results[c][seed]['total_reward'] = total_reward
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as fp:
        json.dump(results, fp)
    print(results)