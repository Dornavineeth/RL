import copy
import numpy as np
from typing import Any
from time import sleep
import gym

import sys
sys.path.append(".")
from environment import get_env


class Node:

    def __init__(self, 
                visits = 0,
                visit_sum = 0.0,
                parent = None,
                observation = None,
                reward = 0,
                terminated = False,
                env = None,
                children = None
                ):
        self.visits = visits
        self.visit_sum = visit_sum
        self.parent = parent
        self.children = children
        self.observation = observation 
        self.reward = reward
        self.terminated = terminated
        self.env = env
    
    def has_children(self):
        return self.children is not None
    

class Selection:
    def __init__(self, config):
        self.config = config
    
    def select_ucb(self, node, *args, **kwargs):
        child_values = {
            action: child.visit_sum/child.visits + self.config['ucb_c'] * np.sqrt(np.log(node.visits)/child.visits) if child.visits>0 else 10000 + np.random.rand(1).item()
            for action, child in node.children.items()
        }
        # Find the maximum value in the dictionary
        max_value = max(child_values.values())
        # Filter actions that have the maximum value
        max_value_actions = [action for action, value in child_values.items() if value == max_value]
        # Randomly sample an action from the list of actions with the maximum value
        action = np.random.choice(max_value_actions)
        child = node.children[action]
        return action, child

    def select_eps_greedy(self, node, *args, **kwargs):
        child_values = {
            action: child.visit_sum/child.visits if child.visits>0 else 10000 + np.random.rand(1).item()
            for action, child in node.children.items()
        }
        if np.random.uniform(0, 1) < self.config['eps']:
            # Explore: choose a random action
            action = np.random.choice(list(child_values.keys()))
        else:
           # Find the maximum value in the dictionary
            max_value = max(child_values.values())
            # Filter actions that have the maximum value
            max_value_actions = [action for action, value in child_values.items() if value == max_value]
            # Randomly sample an action from the list of actions with the maximum value
            action = np.random.choice(max_value_actions)
        child = node.children[action]
        return action, child

    def __call__(self, node, *args: Any, **kwds: Any) -> Any:
        if self.config['selection_strategy'] == 'UCT':
            return self.select_ucb(node, *args, **kwds)
        elif self.config['selection_strategy'] == 'eps-Greedy':
            return self.select_eps_greedy(node, *args, **kwds)

class MCTS:

    def __init__(self, config):
        self.config = config
        self.selection = Selection(config)
        self.debug = self.config.get('debug', False)
    

    def select_child(self, node):
        if node.children is None:
            raise ValueError("select_child is called on nodes with no children :(")
        if node.terminated:
            raise ValueError("Selected is called on a terminal node")
        action, selected_node = self.selection(node)
        # Only for stochastic things
        env = copy.deepcopy(node.env)
        observation, reward, terminated, info = env.step(action)
        selected_node.env = env
        selected_node.reward = reward
        selected_node.observation = observation
        selected_node.terminated = terminated
        return action, selected_node

    def expand(self, node, env):
        node.children = {}
        if node.terminated:
            raise ValueError("Expanding a terminal node")
        for action in range(env.action_space.n):
            env_copy = copy.deepcopy(env)
            observation, reward, terminated, info = env_copy.step(action)
            child_node = Node(
                visits=0,
                visit_sum=0.0,
                parent=node,
                observation=observation,
                reward=0,#reward,
                terminated=False,#terminated,
                env=env_copy,
            )
            assert child_node.children is None, "expanded nodes cant have children"
            node.children[action] = child_node
    
    def rollout(self, env, depth, max_depth):
        reward_rollout = 0
        terminated = False
        action_space = list(range(env.action_space.n))
        rollout_depth = 0 
        while depth<max_depth:
            if terminated:
                break
            action = np.random.choice(action_space)
            observation, reward, terminated, info = env.step(action)
            reward_rollout += ((self.config['discount']**rollout_depth)*reward)
            rollout_depth+=1
            depth+=1
        return reward_rollout


    def back_propogate(self, node, rollout_value):
        if node.terminated:
            node.visit_sum += node.reward # reward on terminal state
            node.visits += 1
            rollout_value = node.reward + (self.config['discount'] * rollout_value)
            node = node.parent
        while node is not None:
            if node is not None and node.terminated:
                raise ValueError("Traj shouldnt have terminal state in between")
            node.visit_sum += rollout_value
            node.visits += 1
            rollout_value = node.reward + (self.config['discount'] * rollout_value)
            node = node.parent
            

    def select_best_action(self, node):
        assert node.children is not None, "node has no children"
        child_values = {
            action: child.visit_sum/child.visits if child.visits > 0 else 0
            for action, child in node.children.items()
        }
        max_value = max(child_values.values())
        # Filter actions that have the maximum value
        max_value_actions = [action for action, value in child_values.items() if value == max_value]

        # Randomly sample an action from the list of actions with the maximum value
        selected_action = np.random.choice(max_value_actions)
        return selected_action

    def _print_root_node_stats(self, root_node):
        print("root_node visits ", root_node.visits)
        print("root_node visit_sum ", root_node.visit_sum)
        for ch in root_node.children:
            print(f"child {ch} visits ", root_node.children[ch].visits)
            print(f"child {ch} visit_sum ", root_node.children[ch].visit_sum)

    def search(self, env, time_step):
        budget = self.config['budget']
        max_depth = self.config['max_depth']
        root_env = copy.deepcopy(env)
        root_node = Node(
            env = root_env
        )

        for it in range(budget):
            if self.debug:
                print(f"Iteration {it} : Started")
            node = root_node
            depth = 0

            # Selection
            while True:
                if not node.has_children() or depth >= max_depth or node.terminated:
                    break
                action, node = self.select_child(node)
                depth += 1

            
            if self.debug:
                print(f"Selection {it} : Done")

            # Expand Tree
            if not node.terminated and depth+1<max_depth:
                env_copy = copy.deepcopy(node.env)
                self.expand(node, env_copy)
                assert len(node.children) == env.action_space.n , "expansion went wrong" 
                expanded = True
                if self.debug:
                    print(f"Expand {it} : Done")
            else:
                expanded = False

            if expanded:
                action, child_node = self.select_child(node)
    
                # RollOut
                if child_node.terminated:
                    rollout_value = 0
                else:
                    rollout_env = copy.deepcopy(child_node.env)
                    rollout_value = self.rollout(rollout_env, depth, max_depth)
                if self.debug:
                    print(f"Rollout {it} : Done")
            else:
                child_node = node
                rollout_value = 0
                
            # BackPropogate
            self.back_propogate(child_node, rollout_value)
            if self.debug:
                print(f"BackPropogate {it} : Done")
        best_action = self.select_best_action(root_node)
        self._print_root_node_stats(root_node)
        return best_action

    
class MCTSAgent:

    def __init__(self, config):
        self.config = config
        self.mcts = MCTS(config)
    
    def step(self, env, time_step):
        action = self.mcts.search(env, time_step)
        return action


def run_episode(config):
    seed = config['seed']
    np.random.seed(seed)
    env_type = config['env_type']
    env = get_env(env_type)
    observation = env.reset()
    env.seed(seed)
    actions_traj = []
    terminated = False
    agent = MCTSAgent(config)
    total_reward = 0
    time_step = 0
    states_traj = []
    while not terminated:
        # env.render()
        states_traj.append(observation)
        action = agent.step(env, time_step)
        actions_traj.append(action)
        observation, reward, terminated, info = env.step(action)
        total_reward = total_reward + (config['discount']**time_step)*reward
        print(action, observation, reward, terminated, info, total_reward)
        time_step+=1
        sleep(0.5)
        if terminated:
            break
    return total_reward, time_step

if __name__ == "__main__":
    config = {
        'env_type' : 'gridworld',
        # 'env_type': "MountainCar-v0",
        'budget': 1000,
        'selection_strategy' : 'UCT',
        'discount': 1.0,
        'ucb_c': 50.0,
        'max_depth': 200,
        'eps': 0.1,
        'debug' : False,
        'seed': 1,
    }
    print(run_episode(config))



