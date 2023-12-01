import copy
import numpy as np
from typing import Any
from time import sleep
# import gymnasium as gym
import gym



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
        # print(child_values)
        action = max(child_values, key=child_values.get)
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
            # Exploit: choose the action with the highest value
            action = max(child_values, key=child_values.get)
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
        action, selected_node = self.selection(node)
        return action, selected_node

    def expand(self, node, env):
        node.children = {}
        for action in range(env.action_space.n):
            env_copy = copy.deepcopy(env)
            observation, reward, terminated, info = env_copy.step(action)
            child_node = Node(
                visits=0,
                visit_sum=0.0,
                parent=node,
                observation=observation,
                reward=reward,
                terminated=terminated,
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
        while node is not None:
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
        action = max(child_values, key=child_values.get)
        return action

    def _print_root_node_stats(self, root_node):
        print("root_node visits ", root_node.visits)
        print("root_node visits ", root_node.visits)

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
        # if time_step ==100:
        #     import pdb;pdb.set_trace()
        # print()
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
    np.random.seed(0)
    env_type = config['env_type']
    env = gym.make(env_type)
    state = env.reset()
    import pdb;pdb.set_trace()
    env.seed(seed)
    env_orig = copy.deepcopy(env)
    actions_traj = []
    terminated = False
    agent = MCTSAgent(config)
    total_reward = 0
    time_step = 0
    while not terminated:
        action = agent.step(env, time_step)
        actions_traj.append(action)
        observation, reward, terminated, info = env.step(action)
        total_reward = total_reward + (config['discount']**time_step)*reward
        print(action, observation, reward, terminated, info, total_reward)
        time_step+=1
        if terminated:
            break

    terminated = False
    for time_step, action in enumerate(actions_traj):
        _, reward, terminated, _ = env_orig.step(action)
        env_orig.render()
        sleep(0.05)
        if terminated:
            print(f"terminated at {time_step}, before finishing all acitons")
            break
    return total_reward

if __name__ == "__main__":
    config = {
        'env_type': 'CartPole-v1',
        # 'env_type' : 'MountainCar-v0',
        # 'env_type': 'FrozenLake-v1',
        'budget': 100,
        'selection_strategy' : 'UCT',
        # 'selection_strategy' : 'eps-Greedy',
        'discount': 0.999,
        'ucb_c': 100.0,
        'max_depth': 50,
        'eps': 0.1,
        'debug' : False,
        'seed': 0,
    }
    print(run_episode(config))
    

    # num_seeds = config['num_seeds']
    # env_type = config['env_type']
    # budgets = config['budgets']
    # reward_per_budget = []

    # env = gym.make(env_type)
    # state = env.reset()
    # env.seed(0)
    # env_orig = copy.deepcopy(env)
    # actions_traj = []
    # done = False
    # agent = MCTSAgent(config)
    # total_reward = 0
    # while not done:
    #     action = agent.step(env)
    #     actions_traj.append(action)
    #     # observation, reward, terminated, truncated, info = env.step(action)
    #     observation, reward, terminated, info = env.step(action)
    #     total_reward = total_reward + config['discount']*reward
    #     # print(observation, reward, terminated, truncated, info)
    #     print(observation, reward, terminated, info, total_reward)
    #     if terminated:
    #         break
    
    # done = False
    # for action in actions_traj:
    #     _, reward, done, _ = env_orig.step(action)
    #     env_orig.render()
    #     if done:
    #         break
    












    # for budget in budgets:
    #     budget_config = copy.deepcopy(config)
    #     budget_config['budget'] = budget

    #     agent = MCTSAgent(budget_config)
    #     env = gym.make(env_type)
    #     reward_per_seed = [] 
    #     for seed in range(num_seeds):
    #         np.random.seed(seed)
    #         state = env.reset(seed=seed)
    #         done = False
    #         total_reward = 0
    #         actions_traj = []
    #         while not done:
    #             action = agent.step(env)
    #             actions_traj.append(action)
    #             observation, reward, terminated, truncated, info = env.step(action)
    #             total_reward = total_reward + config['discount']*reward
    #             # print(observation, reward, terminated, truncated, info)
    #             if terminated or truncated:
    #                 break
    #             env.render()
    #         reward_per_seed.append(total_reward)
    #         print(f"Budget : {budget}, Seed : {seed}, Total reward : {total_reward}")
    #     reward_per_budget.append(reward_per_seed)


