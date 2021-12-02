from agent_dir.agent import Agent
import scipy
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm_notebook

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
        self.NUM_BATCH = 400        # 總共更新 400 次
        
        self.network = PolicyGradientNetwork()
        self.agent = PolicyGradientAgent()


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        avg_total_rewards, avg_final_rewards = [], []

        prg_bar = tqdm_notebook(range(NUM_BATCH))
        for batch in prg_bar:

            log_probs, rewards = [], []
            total_rewards, final_rewards = [], []

            # 蒐集訓練資料
            for episode in range(self.EPISODE_PER_BATCH):

                state = env.reset()
                total_reward, total_step = 0, 0

                while True:

                    action, log_prob = self.agent.sample(state)
                    next_state, reward, done, _ = env.step(action)

                    log_probs.append(log_prob)
                    state = next_state
                    total_reward += reward
                    total_step += 1

                    if done:
                        final_rewards.append(reward)
                        total_rewards.append(total_reward)
                        rewards.append(np.full(total_step, total_reward))  # 設定同一個 episode 每個 action 的 reward 都是 total reward
                        break

            # 紀錄訓練過程
            avg_total_reward = sum(total_rewards) / len(total_rewards)
            avg_final_reward = sum(final_rewards) / len(final_rewards)
            avg_total_rewards.append(avg_total_reward)
            avg_final_rewards.append(avg_final_reward)
            prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

            # 更新網路
            rewards = np.concatenate(rewards, axis=0)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
            agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()

    
    
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),    
            nn.MaxPool2d(2)
        )  
        
        self.seq2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )         
        
        self.fc = nn.Linear(65536, 1)

    def forward(self, state):
        x = self.seq1(state)
        x = self.seq2(x)
        x = self.seq2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
    
class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
