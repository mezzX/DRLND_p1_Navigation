import numpy as np
import random
from collections import namedtuple, deque
from model import Qnetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

buffer_size = int(1e5)  #replay buffer size
batch_size = 64         #minibatch size
gamma = 0.99            #discount factor
tau = 1e-3              #soft update of target params
lr = 5e-4               #learning rate
update_every = 4        #how often to update the network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class agent():


    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #Qnetwork
        self.Qnetwork_local = Qnetwork(state_size, action_size, seed).to(device)
        self.Qnetwork_target = Qnetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.Qnetwork_local.parameters(), lr=lr)

        #replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        #init time step
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        #save step to replay memory
        self.memory.add(state, action, reward, next_state, done)

        #learn every update_every t_steps
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            #check if enough samples are in memory if there are then learn
            if len(self.memory) > batch_size:
                exps = self.memory.sample()
                self.learn(exps, gamma)


    def act(self, state, eps=0.):
        '''Returns actions for a given state based on the current policy
        Params
        ======
            state (array_like): current state
            eps (float) epsilon for epsilon-greedy action selection
        '''

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.Qnetwork_local(state)
        self.Qnetwork_local.train()

        #epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, exps, gamma):
        """Update value parameters using a batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, done = exps

        #get max predicted Q values for next state from target model
        Q_targets_next = self.Qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done))

        #calculate expected Q values from local model
        Q_expected = self.Qnetwork_local(states).gather(1, actions)

        #compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

        #update target network
        self.soft_update(self.Qnetwork_local, self.Qnetwork_target, tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)



class ReplayBuffer():
    '''fixed size buffer to store exps tuples'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.exps = namedtuple('exps', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)


    def add(self, state, action, reward,next_state, done):
        '''add a new exp to memory'''
        e = self.exps(state, action, reward, next_state, done)
        self.memory.append(e)


    def sample(self):
        '''randomly sample a batch of exps from memory'''
        exps = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(device)
        done = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, done)


    def __len__(self):
        '''returns current size of Replay Buffer'''
        return len(self.memory)