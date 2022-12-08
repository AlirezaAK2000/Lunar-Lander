import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class DeepNetwork(nn.Module):
    
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_action) -> None:
        super().__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_action = n_action
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        return output
    

class Agent():
    
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size
                 ,n_action, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4) -> None:
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_action)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.Q_eval = DeepNetwork(n_action=n_action, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256).to(self.device)
        
        self.target_eval = DeepNetwork(n_action=n_action, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256).to(self.device)
        
        self.target_eval.load_state_dict(self.Q_eval.state_dict())
        self.target_eval.eval()
        
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        
        
        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32
        )
        
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32
        )
        
        self.action_memory = np.zeros(
            self.mem_size, dtype=np.int32
        )
        
        self.reward_memory = np.zeros(
            self.mem_size, dtype=np.float32
        )
        
        self.terminal_memory = np.zeros(
            self.mem_size, dtype=bool
        )
        
    def store_transition(self, state, action, reward, state_, done):
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self, observation):
        
        if np.random.random() > self.epsilon:
            with T.no_grad():
                state = T.tensor([observation]).to(self.device)
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()
        
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def learn(self, steps):
        if self.mem_cntr < self.batch_size:
            return
        
        self.optimizer.zero_grad()
        
        max_mem = min(self.mem_size ,self.mem_cntr)
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_eval.forward(new_state_batch).detach()
        
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.loss(q_target, q_eval).to(self.device)
        loss.backward()
        for param in self.Q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        
        if steps % 5 == 0:
            self.target_eval.load_state_dict(self.Q_eval.state_dict())
            
        
        
        
        
        
        
        
        
        
            
        
        