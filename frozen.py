import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
n_steps = 4  # Number of steps for n-step TD-learning
num_episodes = 5000

# Initialize Q-table (16 states × 4 actions for FrozenLake)
Q = np.zeros((env.observation_space.n, env.action_space.n))

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
            
            # Modify reward to encourage progress
            if terminated and reward == 0:
                reward = -1  # Penalize falling into a hole
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
            # Use the ratio of target policy (greedy) to behavior policy (epsilon-greedy)
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
            Q[update_state, update_action] += alpha * rho * (G - Q[update_state, update_action])
        
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
plt.title(f"Off-Policy n-Step TD Learning (n={n_steps}) on FrozenLake")
plt.legend()
plt.grid()
plt.show()

# Print the optimal policy learned
optimal_policy = np.array([np.argmax(Q[s]) for s in range(env.observation_space.n)])
print("Optimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
policy_grid = optimal_policy.reshape(4, 4)
print(policy_grid)

# Visualize the optimal policy
directions = ['L', 'D', 'R', 'U']
policy_chars = np.array([[directions[a] for a in row] for row in policy_grid])
print("\nPolicy Visualization:")
for row in policy_chars:
    print(' '.join(row))

with open("policy_visualization-frozen.txt", "w", encoding="utf-8") as f:
    for row in policy_chars:
        f.write(' '.join(row) + '\n')
print("\nPolicy visualization saved to 'policy_visualization-frozen.txt'")