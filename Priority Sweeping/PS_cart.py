import heapq
import numpy as np
from collections import defaultdict
import gym
from gridworld import TwoDGridWorld
import copy
import random

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.key_index = {}  # key to index mapping
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        _, _, item = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for idx, (p, c, i) in enumerate(self.heap):
            if i == item:
                # item already in, so has either lower or higher priority
                # if already in with smaller priority, don't do anything
                if p <= priority:
                    break
                # if already in with larger priority, update the priority and restore min-heap property
                del self.heap[idx]
                self.heap.append((priority, c, i))
                heapq.heapify(self.heap)
                break
            else:
                # item is not in, so just add to priority queue
                self.push(item, priority)


class BaseAgent:
    def __init__(self, env, gamma=1, epsilon=0.1, alpha=0.5, bucket = 10): 
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.pos_space = np.linspace(-2.4,2.4,bucket)
        self.vel_space = np.linspace(-4,4,bucket)
        self.ang_space = np.linspace(-0.21,0.21,bucket)
        self.angvel_space = np.linspace(-4,4,bucket)


    def reset(self):
        self.q_values = defaultdict(float)
        self.num_updates = 0

    def get_action(self, state):
        rand = np.random.rand()
        actions = range(self.env.action_space.n)
        if rand < self.epsilon:
            act = actions[np.random.choice(len(actions))]
        else:
            act = self.compute_best_action(state)
        return act

    def get_q_value(self, state, action):
        # print(state,action, (action,))
        return self.q_values[(state + (action,))]

    def get_value(self, state):
        best_action = self.compute_best_action(state)
        if best_action is None:
            return 0
        else:
            return self.get_q_value(state, best_action)

    def compute_best_action(self, state): 
        legal_actions = range(self.env.action_space.n)
        if legal_actions[0] is None:
            return None
        q_values_a = [self.get_q_value(state, a) for a in legal_actions]

        
        eligible_best_actions = [a for i, a in enumerate(legal_actions) if np.round(q_values_a[i], 8) == np.round(np.max(q_values_a), 8)]
        best_action_idx = np.random.choice(len(eligible_best_actions))
        best_action = eligible_best_actions[best_action_idx]

        # best_action = legal_actions[np.argmax(q_values_a)]
        # print('best', best_action)
        return best_action

    def get_bucket(self, state):
        x = np.digitize(state[0], self.pos_space)
        v = np.digitize(state[1], self.vel_space)
        w = np.digitize(state[2], self.ang_space)
        w_ = np.digitize(state[3], self.angvel_space)
        return (x,v,w,w_)

    def update(self, state, action, reward, next_state):
        q_t0 = self.get_q_value(state, action)
        q_t1 = self.get_value(next_state)
        new_value = q_t0 + self.alpha * (reward + self.gamma * q_t1 - q_t0)
        self.q_values[(state + (action,))] = new_value
        self.num_updates += 1
        return new_value
    

class Agent(BaseAgent):
    def __init__(self, n_planning_steps, theta, **kwargs):
        super().__init__(**kwargs)
        self.n_planning_steps = n_planning_steps
        self.theta = theta  
        self.reset()

    def reset(self):
        super().reset()
        self.model = {}
        self.pq = PriorityQueue()
        self.predecessors = defaultdict(set)

    def update(self, state, action, reward, next_state):
        state = super().get_bucket(state)
        next_state = super().get_bucket(next_state)
        self.model[state + (action,)] = ((reward,) + next_state)
        self.predecessors[next_state].add((state + (action,)))

        delta = reward + self.gamma * self.get_value(next_state) - self.get_q_value(state, action)
        if abs(delta) > self.theta:
            self.pq.push((state+(action,)), -abs(delta))

        for i in range(self.n_planning_steps):
            if self.pq.is_empty():
                break
            
            tple = self.pq.pop()
            state, action = tple[:4], tple[-1]
            tple2 = self.model[(state + (action,))]
            reward, next_state = tple2[0], tple2[1:]
            # print('nn', next_state)

            super().update(state, action, reward, next_state)
            
            for sa in self.predecessors[state]:
                s, a = sa[:4], sa[-1]
                rx = self.model[(s+(a,))]
                r = rx[0]
                proposed_update = r + self.gamma * self.get_value(state) - self.get_q_value(s, a)
                if abs(proposed_update) > self.theta:
                    self.pq.push((s+(a,)), -abs(proposed_update))


def run_episode(mdp, optimal = False):
    states_visited = []
    actions_performed = []
    episode_rewards = 0
    state = (0,0,0,0)
    states_visited.append(state)
    isTerminal = False
    time_stamp = 0
    env = copy.deepcopy(mdp.env)
    env.reset()
    # print('herhe', mdp.q_values)

    while not isTerminal:
        if optimal:
            action = mdp.compute_best_action(state)
        else:
            action = mdp.get_action(state)
        time_stamp += 1
        next_state, reward, isTerminal,_,_ = env.step(action)
        next_state = tuple(next_state)
        # print(next_state, action, isTerminal)
        mdp.update(state, action, reward, next_state)
        state = next_state

        states_visited.append(state)
        actions_performed.append(action)
        episode_rewards += reward
    # print('total time taken: ', time_stamp)
    return states_visited, actions_performed, episode_rewards


def render(actions_traj, env_orig):
    terminated = False
    for time_step, action in enumerate(actions_traj):
        _, reward, terminated, _ = env_orig.step(action)
        env_orig.render()
        if terminated:
            print(f"terminated at {time_step}, before finishing all acitons")
            break


def get_optimal_policy(q_table):
    optimal_policy = {}
    for state_action, q_value in q_table.items():
        state, action = state_action
        if state not in optimal_policy or q_value > q_table[(state, optimal_policy[state])]:
            optimal_policy[state] = action

    return optimal_policy

if __name__ == '__main__':
    env_type = 'CartPole-v1' 
    n = 50                    # Number of planning steps 
    theta = 0.01                # Threshold

    env_orig = gym.make(env_type)
    mdp = Agent(env=env_orig, n_planning_steps=n, theta=theta, alpha=0.4, epsilon=0.15, gamma=1, bucket =20)
    episodic_rewards = []
    total_reward = 0
    i = 0
    while(total_reward < 500):
        i += 1
        states_visited,actions_performed,total_reward = run_episode(mdp)
        if (i%100 == 0):
            print("total_reward, episode",i,": ", total_reward)
        episodic_rewards.append(total_reward)


    # print('Avg return', np.mean(episodic_rewards))

    # _,_,total_reward = run_episode(mdp, True)
    # print("total_reward, optimal",i,": ", total_reward)

    # q_table = mdp.q_values
    # optimal_policy = get_optimal_policy(q_table)
    # print(optimal_policy)


