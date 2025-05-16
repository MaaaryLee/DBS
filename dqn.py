import torch
from torch import nn 
import torch.nn.functional as func
from collections import deque
import random
from BGN_RL import BGN_RL
import numpy as np
import yaml
import os
import matplotlib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda')


DATE_FORMAT = '%m-%d %H:%M:%S'
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

        
class Dueling_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc_h_v = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.fc_z_v = nn.Linear(hidden_size, 1, dtype=torch.float64)
        self.fc_h_a = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.fc_z_a = nn.Linear(hidden_size, output_size, dtype=torch.float64)

    def forward(self, x):
        v = func.tanh(self.fc_h_v(x))
        v = self.fc_z_v(v)
        a = func.tanh(self.fc_h_a(x))
        a = self.fc_z_a(a)
        return (v + a-a.mean())

# class Dueling_DQN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 16), stride=4)
#         self.conv2 = nn.Conv2d(32, 8, (1, 64), 2)
#         self.conv3 = nn.Conv2d(8, 1, (1, 8), 4)
#         self.fc_h_v = nn.Linear(input_size, hidden_size, dtype=torch.float64)
#         self.fc_z_v = nn.Linear(hidden_size, 1, dtype=torch.float64)
#         self.fc_h_a = nn.Linear(input_size, hidden_size, dtype=torch.float64)
#         self.fc_z_a = nn.Linear(hidden_size, output_size, dtype=torch.float64)

#     def forward(self, x):
#         x = x.type(torch.float32).unsqueeze(1)
#         x = func.gelu(self.conv1(x))
#         x = func.relu(self.conv2(x))
#         x = func.relu(self.conv3(x)).type(torch.float64).squeeze(1)
#         v = func.tanh(self.fc_h_v(x))
#         v = self.fc_z_v(v)
#         a = func.tanh(self.fc_h_a(x))
#         a = self.fc_z_a(a)
#         return (v + a-a.mean()).squeeze(1)

#         # x = self.l1(x)
#         # x = func.tanh(x)
#         # x = self.l2(x)
#         # x = func.relu(x)
#         # x = self.l3(x).squeeze(1)
#         return x
        
    
class ReplayMemory():
    def __init__(self, maxlen):
        self.mem = deque([], maxlen=maxlen)

    def append(self, transition):
        self.mem.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.mem, sample_size)
    
    def __len__(self):
        return len(self.mem)
    


class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']    
        self.mini_batch_size    = hyperparameters['mini_batch_size']        
        self.epsilon_init       = hyperparameters['epsilon_init']          
        self.epsilon_decay      = hyperparameters['epsilon_decay']          
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.sync_rate          = hyperparameters['sync_rate']
        self.discount_factor    = hyperparameters['discount_factor']
        self.lr                 = hyperparameters['lr']
        self.hidden_size        = hyperparameters['hidden_size']

        self.optimizer = None
        self.criterion = nn.MSELoss()

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        num_states = 2
        num_actions = 57
        num_episodes = 1500
        rewards_per_episode = []
        
        policy_dqn = Dueling_DQN(num_states, self.hidden_size, num_actions).to(device)
        
        if is_training: 
            mem = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            epsilon_history = []
            target_dqn = Dueling_DQN(num_states, self.hidden_size, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.lr)
            best_reward = -99999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            
        for episode in range(num_episodes):
            env = BGN_RL(1, tmax=1500, n=10)
            for _ in range(500): env.step(0)
            state = env.state
            state = torch.tensor(state, dtype=torch.float64, device=device)
            terminated = False
            episode_reward = 0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.sample_actions()
                    action = torch.tensor(action, dtype=torch.float64, device=device)
                else:
                    action = torch.tensor(env.action_space[policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()], dtype=torch.float64, device=device)

                new_state, reward, terminated = env.step(action.item(), 20000)
                new_state = torch.tensor(new_state, dtype=torch.float64, device=device)
                reward = torch.tensor(reward, dtype=torch.float64, device=device)
                episode_reward += reward

                if is_training: 
                    mem.append((state, action, new_state, reward, terminated))
                    step_count += 1

                    if step_count > self.sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                state = new_state

            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            rewards_per_episode.append(episode_reward)
            epsilon_history.append(epsilon)

            if is_training:

                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(mem) > self.mini_batch_size:
                    mini_batch = mem.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # if step_count > self.sync_rate:
                    #     target_dqn.load_state_dict(policy_dqn.state_dict())
                    #     step_count = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)

        actions = torch.stack(actions).type(torch.int64)
        actions_index = []
        action_space = [0, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
        for action in actions: actions_index.append(action_space.index(action.item()))
        action_index = torch.tensor(actions_index, dtype=torch.int64, device=device)

        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float64).float().to(device)
        with torch.no_grad():
            target_q = rewards + (1-terminations)*self.discount_factor*target_dqn(new_states).max(dim=1)[0]
        current_q = policy_dqn(states).gather(dim=1, index=action_index.unsqueeze(dim=1)).squeeze()
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)
        for i in range(len(rewards_per_episode)):
            rewards_per_episode[i] = rewards_per_episode[i].cpu()
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
                



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False)

        
        