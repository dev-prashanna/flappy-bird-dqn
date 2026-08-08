from flappyenv import FlappyBirdEnv
from double_dqn import DoubleDQN
import torch
import pygame


def main():
    EPISODES = 2000
    TARGET_UPDATE = 10
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    env = FlappyBirdEnv()
    agent = DoubleDQN(state_dim=5, action_dim=2)
    total_steps = 0
    best_reward = float("-inf")

    print("Starting training...")
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
            agent.train_step()
            total_steps += 1

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), "best_model.pth")

        print(f"Episode: {episode}, Reward: {episode_reward:.1f}, Epsilon: {epsilon:.4f}, Best: {best_reward:.1f}")

    print("Training completed.")
    pygame.quit()


if __name__ == "__main__":
    main()
