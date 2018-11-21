import random
import numpy as np
from replay_buffer import ReplayBuffer
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent:
    def __init__(
        self, 
        state_size,
        action_size,
        batch_size=64,
        buffer_size=100000,
        gamma=0.99,
        tau=0.001,
        lr=0.0005,
        update_step_interval=4,
        seed=None,
        device=None
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_step_interval = update_step_interval
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.device = device
        
        if seed is not None:
            random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed=seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed=seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size=action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            seed=seed, 
            device=self.device
        )

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step - (self.t_step + 1) % self.update_step_interval
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    

    def act(self, state, epsilon=0):
        """ 
        Uses current policy to return 
        an action based on state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     


    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)