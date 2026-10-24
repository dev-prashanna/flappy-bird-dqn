import sys
from datetime import datetime

from flappyenv import FlappyBirdEnv
from double_dqn import DoubleDQN
import torch
import numpy as np


TRAIN_EVERY = 4
EARLY_STOP_PATIENCE = 50


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def main():
    EPISODES = 3000
    TARGET_UPDATE = 10
    epsilon = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.01

    env = FlappyBirdEnv()
    agent = DoubleDQN(state_dim=5, action_dim=2, batch_size=64)
    total_steps = 0
    best_score = 0
    patience_counter = 0

    log_file = open("training_log.txt", "w")
    sys.stdout = Logger(log_file)

    print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {agent.device}")
    print(f"Episodes: {EPISODES}, Epsilon decay: {epsilon_decay}")
    print("-" * 60)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % TRAIN_EVERY == 0:
                agent.train_step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        score = env.score
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(agent.policy_net.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {score}, Reward: {episode_reward:.1f}, Epsilon: {epsilon:.4f}, Best: {best_score}, Steps: {total_steps}")

        if patience_counter >= EARLY_STOP_PATIENCE and best_score > 0:
            print(f"Early stopping at episode {episode} (best score: {best_score})")
            break

    print("-" * 60)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total steps: {total_steps}, Best score: {best_score}")
    torch.save(agent.policy_net.state_dict(), "final_model.pth")
    print("Saved final model to final_model.pth")

    sys.stdout = sys.stdout.terminal
    log_file.close()


if __name__ == "__main__":
    main()
