#coding:utf-8
# Authors: Junior Costa de Jesus #
import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from environment_stage_1 import Env
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gc
import torch.nn as nn
import math
from collections import deque
import copy
from torch.optim import Adam

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

#****************************************************
## 一个环形回放缓存，用于存储转移数据并提供数据采样
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    #随即采样数据
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
# 初始化目标网络时所需的硬拷贝更新
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
 # 更新目标网络时所用到的软更
# 新，使用了 Polyak 平均
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

 # 更新 SAC 中所有的网络
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action
## 用于评估状态-动作值 Q(s,a) 的网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        x_state_action = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear4_q1(x1)
        
        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear4_q2(x2)
        
        return x1, x2

class PolicyNetwork(nn.Module):
    # # 初始化
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
# 前向传播
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std
# 采样动作
    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()# 评估时不进行裁剪，裁剪会影响梯度
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)# 动作选用 TanhNormal 分布；这里使用了重参数技术
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # normal.log_prob 和 -log(1-a**2) 的维度都是 (N,dim_of_action);
        # Normal.log_prob 输出了和输入特征一样的维度，而不是 1 维的概率
        # 这里需要跨维度相加，来得到 1 维的概率，或者使用多元正态分布
        # 由于 reduce_sum 减少了 1 个维度，这里将维度扩展回来
        return action, log_prob, mean, log_std

class SAC(object):
    def __init__(self, state_dim,
                 action_dim, gamma=0.99, 
                 tau=1e-2, 
                 alpha=0.2, 
                 hidden_dim=256,
                 lr=0.0003):
# 建立网络及变量
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr=lr
        self.target_update_interval = 1
        self.q_loss =0.0
        self.policy_loss=0.0
        self.alpha_loss =0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        print('entropy', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        
        # 输出动作
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)# 动作分布使用 TanhNormal 分布; 这里使用了重参数技术
        action = action.detach().cpu().numpy()[0]
        return action
    #获取训练数据
    def update_parameters(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # 扩展维度
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        # 通过批数据的均值和标准差进行标准化，并增加一个极小的数防止除以 0 导致数值溢出问题
    # 进行评估
        with torch.no_grad():
            # 训练 Q 函数
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # 如果 done==1，则只有 reward 值
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # 
        qf2_loss = F.mse_loss(qf2, next_q_value) # 
        qf_loss = qf1_loss + qf2_loss
        self.q_loss =qf_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        # 训练策略网络
        pi, log_pi, mean, log_std = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # 更新 alpha
        # alpha: 探索（最大化熵）和利用（最大化 Q 值）之间的权衡
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_loss =policy_loss 
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # 固定 alpha 值
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_loss =alpha_loss
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        soft_update(self.critic_target, self.critic, self.tau)
        #保存模型
    def save_models(self, episode_count):
        torch.save(self.policy.state_dict(),dirPath+'/model/'+str(episode_count)+ '_policy_net.pth')
        torch.save(self.critic.state_dict(), dirPath  +  '/model/'+str(episode_count)+ 'value_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")
    #加载模型
    def load_models(self, episode):
        self.policy.load_state_dict(torch.load(dirPath + '/model/' +str(episode)+ '_policy_net.pth'))
        self.critic.load_state_dict(torch.load(dirPath + '/model/' +str(episode)+ 'value_net.pth'))
        hard_update(self.critic_target, self.critic)
        print('***Models load***')


is_training = True

max_episodes  = 10001
max_steps   = 5000
rewards     = []
batch_size  = 256

action_dim = 2
state_dim  = 28
hidden_dim = 500
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -1. # rad/s
ACTION_V_MAX = 0.3 # m/s
ACTION_W_MAX = 2. # rad/s
world = 'stage_1'
replay_buffer_size = 50000

agent = SAC(state_dim, action_dim)
replay_buffer = ReplayBuffer(replay_buffer_size)
# agent.load_models(445)


print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')


#训练
if __name__ == '__main__':
    rospy.init_node('sac')
    pub_result = rospy.Publisher('result',  Float32MultiArray, queue_size=5)
    print('start trainning!!!')
    result =  Float32MultiArray()
    env = Env()
    before_training = 4
    past_action = np.array([0.,0.])
    for ep in range(max_episodes):
        done = False
        # 这里需要进行一次额外的调用，来使内部函数进行一些初始化操作，让其可以正常使用
        # model.forward 函数
        state = env.reset()
        
        if is_training and not ep%10 == 0 and len(replay_buffer) > before_training*batch_size:
            print('Episode: ' + str(ep) + ' training')
        else:
            if len(replay_buffer) > before_training*batch_size:
                print('Episode: ' + str(ep) + ' evaluating')
            else:
                print('Episode: ' + str(ep) + ' adding to memory')

        rewards_current_episode = 0.

        for step in range(max_steps):
            state = np.float32(state)
            # print('state___', state)
            if is_training and not ep%10 == 0:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, eval=True)

            if not is_training:
                action = agent.select_action(state, eval=True)
            unnorm_action = np.array([action_unnormalized(action[0], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)])

            next_state, reward, done = env.step(unnorm_action, past_action)
            # print('action', unnorm_action,'r',reward)
            past_action = copy.deepcopy(action)

            rewards_current_episode += reward
            next_state = np.float32(next_state)

            if not ep%10 == 0 or not len(replay_buffer) > before_training*batch_size:
                if reward == 100.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        replay_buffer.push(state, action, reward, next_state, done)
                else:
                    replay_buffer.push(state, action, reward, next_state, done)
            
            if  is_training and len(replay_buffer) > batch_size:
                agent.update_parameters(replay_buffer, batch_size)
                print('update_parameters')
                
            state = copy.deepcopy(next_state)

            if done:
                break
        
        print('reward per ep: ' + str(rewards_current_episode))
        print('reward average per ep: ' + str(rewards_current_episode) + ' and break step: ' + str(step))
        if ep%10 == 0:
            if len(replay_buffer) >batch_size:
                result.data = [rewards_current_episode,agent.q_loss,agent.policy_loss,agent.alpha_loss]
                pub_result.publish(result)
                print('pub_result')
        
        if ep%5 == 0:
            agent.save_models(ep)

