import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from environment_stage_4 import Env

import time
import rospy
from std_msgs.msg import Float32MultiArray
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


#Use Cuda
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
#Replay Buffer
from collections import deque
class ReplayBuffer(object):
    def __init__(self):
        self.buffer = deque(maxlen =1000000)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

#Epsilon greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
#Deep Q Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_4_')
        self.result = Float32MultiArray()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_size)
        return action
#input data
action_size =5
state_size =28
model = DQN(state_size, action_size)


if USE_CUDA:
    model = model.cuda()
    
optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer()

#Computing Temporal Difference Loss
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss,q_value
#Training
if __name__=='__main__':
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    num_frames = 10000
    batch_size = 64
    gamma      = 0.99
    action_size =5
    state_size =28
    #agent =DQN(state_size, action_size)
    model = DQN(state_size, action_size)
    losses = []
    all_rewards = []
    episode_reward = 0
    scores,episodes =[],[]
    global_step =0
    start_time =time.time()
    env = Env(action_size)
    EPISODES = 3000
    for e in range(0,  EPISODES):
        done=False
        state = env.reset()
        epsilon = epsilon_by_frame(500)
        # action = model.act(state, epsilon)
        score =0
        episode_step=6000
#training
        for t in range(episode_step):
            action = model.act(state, epsilon)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer)  >=batch_size:
                if global_step<=2000:
                    compute_td_loss(batch_size)
                else:
                    compute_td_loss(batch_size)

            score+=reward
            state=next_state
            get_action.data=[action, score,reward]
            pub_get_action.publish(get_action)
            episode_reward += reward

            if e % 10 == 0:
                torch.save(DQN,str(e)+'save.pt')
            if t >=250:
                rospy.loginfo("time out!")
                done =True

            if done:
            # state = env.reset()
            # all_rewards.append(episode_reward)
            # episode_reward = 0
                result.data=[score]
                #print("score:"+ result.data)
                pub_result.publish(result)
                #score.append(score)
                episodes.append(e)
                m,s =divmod(int(time.time()- start_time),60)
                h,m =divmod(m,60)
                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(replay_buffer), epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break
            global_step +=1
            if global_step % 2000==0:
                rospy.loginfo("update target network")
        if epsilon_by_frame >epsilon_final:
           epsilon = epsilon_by_frame(e)
            
