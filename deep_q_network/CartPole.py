import wandb

from DQNAgent import DQNAgent
import torch
import gymnasium as gym
import numpy as np


def train_agent(agent):
    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    # Logging
    wandb.init(project=agent.cfg.wandb_name, config=agent.cfg.get_members())

    for episode in range(1, agent.cfg.episodes + 1):

        if episode == int(agent.cfg.episodes / 4):
            agent.cfg.epsilon = agent.cfg.epsilon / 2

        if episode == int((agent.cfg.episodes / 2) + (agent.cfg.episodes / 4)):
            agent.cfg.epsilon = agent.cfg.epsilon / 2

        obs, info = train_env.reset()
        episode_return = 0
        episode_lenght = 0
        losses = []

        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = train_env.step(action)

            episode_return += reward
            episode_lenght += 1

            # save transition
            agent.store_transition(ob=obs, ac=action, rew=reward, next_ob=next_obs, done=truncated or terminated)
            # update DQN
            loss = agent.update_q_values()

            if loss is not None:
                losses.append(loss.item())

            obs = next_obs

            if terminated or truncated:
                break

        if not losses:
            wandb.log({"training return": episode_return, "train episode length": episode_lenght,
                       "Epsilon": agent.cfg.epsilon})
        else:
            losses = np.average(losses)
            wandb.log({"training return": episode_return,
                       "train episode length": episode_lenght,
                       "loss": losses,
                       "Epsilon": agent.cfg.epsilon})

        print("Episode", episode, "episode return:", episode_return, end="\t")

        # Update the target network
        if episode % agent.cfg.update_target_network_freq == 0:
            agent.update_target_network()

        if episode % agent.cfg.eval_freq == 0:

            obs, info = eval_env.reset()
            episode_return = 0
            episode_lenght = 0

            while True:
                action = agent._greedy_action(obs)
                next_obs, reward, terminated, truncated, info = eval_env.step(action)

                obs = next_obs

                episode_return += reward
                episode_lenght += 1

                if terminated or truncated:
                    break
            wandb.log({"eval return": episode_return, "eval episode length": episode_lenght})
            print("Eval return:", episode_return, end="")
        print()
    wandb.finish()


if __name__ == '__main__':
    agent = DQNAgent(DQNAgent.Config())
    train_agent(agent)

    agent.save("testdqn.pyt")
