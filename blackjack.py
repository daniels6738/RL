import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Create the Blackjack environment
env = gym.make("Blackjack-v1", sab=True)

# Adjusted Hyperparameters
alpha = 0.01  # Smaller learning rate for stability
gamma = 0.9   # Lower discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Higher minimum exploration
epsilon_decay = 0.9995  # Slower decay
n_steps = 3  # Shorter n-step
num_episodes = 20000  # More episodes

# Initialize Q-table
Q = np.zeros((32, 11, 2, 2))  # Player sum, dealer card, usable ace, actions

def state_to_index(state):
    """Convert Blackjack state tuple to indices for Q-table"""
    player_sum, dealer_card, usable_ace = state
    return (player_sum, dealer_card, int(usable_ace))

def epsilon_greedy_policy(state):
    """Returns probabilities for each action under epsilon-greedy policy."""
    probs = np.ones(env.action_space.n) * epsilon / env.action_space.n
    probs[np.argmax(Q[state_to_index(state)])] += 1 - epsilon
    return probs

def greedy_policy(state):
    """Returns probabilities for each action under greedy policy."""
    probs = np.zeros(env.action_space.n)
    probs[np.argmax(Q[state_to_index(state)])] = 1.0
    return probs

def select_action(state, policy_fn):
    """Selects action based on provided policy function."""
    probs = policy_fn(state)
    return np.random.choice(np.arange(len(probs)), p=probs)

# Track metrics
reward_history = []
wins = 0
draws = 0
losses = 0

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    # Initialize storage for n-step updates
    states = deque([state])
    actions = deque()
    rewards = deque()
    behavior_probs = deque()
    
    # Initial action
    action = select_action(state, epsilon_greedy_policy)
    
    t = 0
    T = float('inf')
    
    while True:
        if t < T:
            # Take action, observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Adjust reward structure
            if terminated:
                if reward == 1.0:    # Win
                    wins += 1
                elif reward == 0.0:   # Draw
                    draws += 1
                    reward = 0.0
                else:                 # Loss
                    losses += 1
                    reward = -1.0
                    
            episode_reward += reward
            
            # Store transition
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            # Store behavior policy probability
            behavior_probs.append(epsilon_greedy_policy(state)[action])
            
            if terminated or truncated:
                T = t + 1
            else:
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
                G += (gamma ** n_steps) * np.max(Q[state_to_index(bootstrap_state)])
            
            # Calculate importance sampling ratio
            rho = 1.0
            for i in range(min(n_steps, T - update_time)):
                state_i = states[i]
                action_i = actions[i]
                target_prob = greedy_policy(state_i)[action_i]
                behavior_prob = behavior_probs[i]
                
                if behavior_prob > 0:
                    rho *= target_prob / behavior_prob
                else:
                    rho = 0
                    break
            
            # Update Q value
            update_state = states.popleft()
            update_action = actions.popleft()
            rewards.popleft()
            behavior_probs.popleft()
            
            state_idx = state_to_index(update_state)
            Q[state_idx][update_action] += alpha * rho * (G - Q[state_idx][update_action])
        
        if update_time == T - 1:
            break
        
        t += 1
    
    reward_history.append(episode_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Print progress
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(reward_history[-100:])
        win_rate = wins / (episode + 1)
        print(f"Episode {episode+1}/{num_episodes}, "
              f"Avg Reward (last 100): {avg_reward:.2f}, "
              f"Win Rate: {win_rate:.2%}, "
              f"Epsilon: {epsilon:.3f}")

env.close()

# Plot metrics
plt.figure(figsize=(12, 5))

# First subplot - Rewards with exponential moving average for smoother curves
plt.subplot(1, 2, 1)
# Raw data plot with very low alpha for context
plt.plot(reward_history, label="Raw Rewards", alpha=0.1, color='blue')

# Multiple moving averages for better visualization
window_sizes = [100, 500, 1000]
colors = ['yellow', 'orange', 'red']
for window, color in zip(window_sizes, colors):
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(reward_history)), moving_avg, 
            label=f'{window}-Episode MA', color=color, linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards over Time")
plt.legend()
plt.grid(True, alpha=0.3)

# Second subplot - Win Rates
plt.subplot(1, 2, 2)
window_sizes = [1000, 2000, 5000]
colors = ['lightgreen', 'green', 'darkgreen']

for window, color in zip(window_sizes, colors):
    if len(reward_history) > window:
        # Fixed win rate calculation
        win_rates = []
        for i in range(len(reward_history) - window):
            wins_in_window = sum(1 for r in reward_history[i:i+window] if r > 0)
            win_rates.append(wins_in_window / window)
        
        plt.plot(range(len(win_rates)), win_rates,
                label=f'{window}-Episode WR', color=color, linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.title("Win Rate over Time")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test the learned policy
def test_policy(n_tests=100):
    test_rewards = []
    test_wins = 0
    
    for _ in range(n_tests):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = np.argmax(Q[state_to_index(state)])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if terminated and reward == 1.0:
                test_wins += 1
                
            total_reward += reward
            state = next_state
        
        test_rewards.append(total_reward)
    
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Win Rate: {test_wins/n_tests:.2%}")
    print(f"Test Rewards: {test_rewards}")

print("\nTesting the learned policy...")
test_policy()