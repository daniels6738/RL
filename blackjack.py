import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Create the Blackjack environment
env = gym.make("Blackjack-v1", sab=True)  # Use the "sab" version for Stick-And-Bust rules

# Adjusted Hyperparameters
alpha = 0.05  # Reduced learning rate for more stable updates
gamma = 0.95  # Slightly lower discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.05  # Higher minimum exploration rate to encourage exploration
epsilon_decay = 0.999  # Slower decay for more exploration
n_steps = 6  # Increased number of steps for n-step TD-learning
num_episodes = 10000  # Increased number of episodes for better training

# Initialize Q-table
Q = np.zeros((32, 11, 2, 2))  # Player sum (0-31), dealer card (1-10), usable ace (0/1), actions (stick/hit)

# Track rewards for plotting
reward_history = []

def epsilon_greedy_policy(state):
    """Returns probabilities for each action under epsilon-greedy policy."""
    probs = np.ones(env.action_space.n) * epsilon / env.action_space.n
    probs[np.argmax(Q[state])] += 1 - epsilon
    return probs

def greedy_policy(state):
    """Returns probabilities for each action under greedy policy."""
    probs = np.zeros(env.action_space.n)
    probs[np.argmax(Q[state])] = 1.0
    return probs

def select_action(state, policy_fn):
    """Selects action based on provided policy function."""
    probs = policy_fn(state)
    return np.random.choice(np.arange(len(probs)), p=probs)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    # Initialize storage for n-step updates
    states = deque([state])
    actions = deque()
    rewards = deque()
    behavior_probs = deque()  # Store behavior policy probabilities
    
    # Initial action
    action = select_action(state, epsilon_greedy_policy)
    
    t = 0  # Time step counter
    T = float('inf')  # Terminal time (will be updated when episode ends)
    
    while True:
        if t < T:
            # Take action, observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Adjust reward to penalize losing more heavily
            if terminated and reward == 0:
                reward = -1  # Penalize losing
            episode_reward += reward
            
            # Store transition
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            # Store behavior policy probability for importance sampling
            behavior_probs.append(epsilon_greedy_policy(state)[action])
            
            # Check if episode ended
            if terminated or truncated:
                T = t + 1
            else:
                # Select next action
                action = select_action(next_state, epsilon_greedy_policy)
            
            state = next_state
        
        # Time to update 
        update_time = t - n_steps + 1
        if update_time >= 0:
            # Calculate n-step return
            G = 0
            for i in range(update_time, min(update_time + n_steps, T)):
                step_idx = i - update_time
                G += (gamma ** step_idx) * rewards[step_idx]
            
            # Add bootstrap value if not at terminal state
            if update_time + n_steps < T:
                bootstrap_state = states[n_steps]
                G += (gamma ** n_steps) * np.max(Q[bootstrap_state])
            
            # Calculate importance sampling ratio (for off-policy correction)
            rho = 1.0
            for i in range(min(n_steps, T - update_time)):
                state_i = states[i]
                action_i = actions[i]
                target_prob = greedy_policy(state_i)[action_i]
                behavior_prob = behavior_probs[i]
                
                # Avoid division by zero
                if behavior_prob > 0:
                    rho *= target_prob / behavior_prob
                else:
                    rho = 0
                    break
            
            # Update Q value with importance sampling correction
            update_state = states.popleft()
            update_action = actions.popleft()
            rewards.popleft()
            behavior_probs.popleft()
            
            # Off-policy update with importance sampling
            Q[update_state][update_action] += alpha * rho * (G - Q[update_state][update_action])
        
        if update_time == T - 1:
            break
        
        t += 1
    
    reward_history.append(episode_reward)
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Print progress occasionally
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(reward_history[-100:])
        print(f"Episode {episode+1}/{num_episodes}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# Plot rewards over time
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Episode Reward", alpha=0.6)
plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'), 
         label="100-Episode Moving Average", color='red')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"Off-Policy n-Step TD Learning (n={n_steps}) on Blackjack")
plt.legend()
plt.grid()
plt.show()

# Test the learned policy
def test_policy(n_tests=100):
    test_rewards = []
    
    for _ in range(n_tests):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select greedy action
            action = np.argmax(Q[state])
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
        
        test_rewards.append(total_reward)
    
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Test Rewards: {test_rewards}")

print("\nTesting the learned policy...")
test_policy()