import gymnasium as gym
from itertools import count
from pandas._config import display
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import Agent

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAU = 0.005

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        state, info = self.env.reset()
        self.state = state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        self.observation = None
        self.n_observations = len(state)

    def get_action_space(self):
        return self.env.action_space.n

    def get_environment(self):
        return self.env

    def get_state(self):
        return self.state

    def get_observations(self):
        return torch.tensor(self.observation, dtype=torch.float32, device=device).unsqueeze(0)

    def perform_action(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action.item())
        self.observation = observation
        return observation, reward, terminated, truncated


def main():
    num_episodes = 50
    steps_done = 0
    # Initialize the environment and get it's state
    environment = Environment()
    state = environment.get_state()
    agent = Agent(capacity=10000, n_observations=len(state), n_actions=environment.get_action_space())

    for i_episode in range(num_episodes):
        state = environment.get_state()
        for t in count():
            action = agent.select_action(environment.get_environment(), state, steps_done)
            observation, reward, terminated, truncated = environment.perform_action(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = environment.get_observations()

            # Store the transition in memory
            agent.add_to_memory(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.learn()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            policy_net, target_net = agent.get_networks()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
