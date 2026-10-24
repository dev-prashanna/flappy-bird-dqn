import torch.optim as optim
from neural_net import DQN
import torch.nn as nn
from replay_buffer import ReplayBuffer
import random
import torch

device = torch.device("cpu")


class DoubleDQN:
    def __init__(self, state_dim=5, action_dim=2, lr=0.0005, gamma=0.99, batch_size=128):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=100000)

    def load(self, path):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        self.policy_net.eval()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice([0, 1])

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))

        loss = self.loss_fn(current_q, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
