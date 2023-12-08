import os
import sys
sys.path.append(".")
import json
from copy import deepcopy

from exp.utils import load_config
from agents.mcts_gridworld import run_episode

if __name__ == "__main__":
    path = 'configs/mcts/gridworld_depths.yml'
    out_path = 'results/mcts/gridworld_depths.json'
    out_dir = os.path.dirname(out_path)
    config = load_config(path)

    max_depths = config['max_depths']
    seeds = list(range(config['num_seeds']))

    results = {}
    for max_depth in max_depths:
        if max_depth not in results:
            results[max_depth] = {}
        for seed in seeds:
            results[max_depth][seed] = {} 
            cfg = deepcopy(config)
            cfg['max_depth'] = max_depth
            cfg['seed'] = seed

            total_reward = run_episode(cfg)
            print(f"Max Depth : {max_depth}, Seed : {seed}, Total Reward : {total_reward}")
            results[max_depth][seed]['total_reward'] = total_reward[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as fp:
        json.dump(results, fp)
    print(results)
