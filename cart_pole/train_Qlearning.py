from QLearning import QLearningAgent
import wandb
import gymnasium as gym
import numpy as np


def discretize(state):
    return (
        round(state[0], 1),
        round(state[1], 1),
        round(state[2], 2),
        round(state[3], 1),
    )


def train_agent(agent):
    # Make envs
    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    # Logging
    wandb.init(project=agent.cfg.wandb_name, config=agent.cfg.get_members())

    for episode in range(1, agent.cfg.episodes + 1):

        obs, info = train_env.reset()
        episode_return = 0
        episode_lenght = 0

        while True:
            action = agent.act(discretize(obs))
            next_obs, reward, terminated, truncated, info = train_env.step(action)

            # Update agent
            agent.update_q_values(discretize(obs), reward, action, discretize(next_obs))

            obs = next_obs

            episode_return += reward
            episode_lenght += 1

            if terminated or truncated:
                break

        wandb.log({"training return": episode_return, "train episode length": episode_lenght})
        if i % agent.cfg.print_freq == 0:
            print("Episode", episode, "episode return:", episode_return, end="\t")

        if episode % agent.cfg.eval_freq == 0:

            obs, info = eval_env.reset()
            episode_return = 0
            episode_lenght = 0

            while True:
                action = agent._greedy_action(discretize(obs))
                next_obs, reward, terminated, truncated, info = eval_env.step(action)

                obs = next_obs

                episode_return += reward
                episode_lenght += 1

                if terminated or truncated:
                    break
            wandb.log({"eval return": episode_return, "eval episode length": episode_lenght})
            print("Eval return:", episode_return, end="")
        print("Epsilon: ", agent.epsilon, end="")
        print()
        agent.decay_epsilon(episode)
        agent.decay_lr(episode)

    wandb.finish()


def epsilon_decay(epsilon, i):
    num_episodes = QLearningAgent.Config.episodes
    if i < (num_episodes / 10):
        return 0.9
    if i < (num_episodes / 6):
        return 0.6
    if i < (num_episodes / 4):
        return 0.2
    if i < (num_episodes / 2):
        return 0.1
    else:
        return 0.01


if __name__ == '__main__':
    agent = QLearningAgent(QLearningAgent.Config(), epsilon_decay=epsilon_decay)
    # agent.load(str(agent.cfg.env) + ".agent")
    train_agent(agent)
    agent.save(str(agent.cfg.env) + ".agent")
