import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Create the Acrobot environment
env = gym.make("Acrobot-v1")

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
n_steps = 4  # Number of steps for n-step TD-learning
num_episodes = 2000

# State space discretization for Acrobot
# Acrobot has 6 continuous state variables:
# cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot
n_bins = 10  # Number of bins for each state dimension
state_bounds = [
    [-1.0, 1.0],  # cos(theta1)
    [-1.0, 1.0],  # sin(theta1)
    [-1.0, 1.0],  # cos(theta2)
    [-1.0, 1.0],  # sin(theta2)
    [-12.567, 12.567],  # theta1_dot
    [-28.274, 28.274]   # theta2_dot
]

# Initialize Q-table with discretized state space
Q = np.zeros((n_bins, n_bins, n_bins, n_bins, n_bins, n_bins, env.action_space.n))

# Track metrics for plotting
reward_history = []
episode_length_history = []
average_q_values = []
max_q_values = []

def discretize_state(state):
    """Convert continuous state to discrete state bins."""
    discrete_state = []
    for i, (s, bounds) in enumerate(zip(state, state_bounds)):
        bin_width = (bounds[1] - bounds[0]) / n_bins
        # Clip state to be within bounds
        s = max(bounds[0], min(s, bounds[1]))
        # Calculate bin index
        bin_idx = min(n_bins - 1, int((s - bounds[0]) / bin_width))
        discrete_state.append(bin_idx)
    return tuple(discrete_state)

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
    discrete_state = discretize_state(state)
    episode_reward = 0
    episode_length = 0
    
    states = deque([discrete_state])
    actions = deque()
    rewards = deque()
    behavior_probs = deque()
    
    action = select_action(discrete_state, epsilon_greedy_policy)
    
    t = 0
    T = float('inf')
    
    while True:
        if t < T:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_discrete_state = discretize_state(next_state)
            episode_reward += reward
            episode_length += 1
            
            actions.append(action)
            rewards.append(reward)
            states.append(next_discrete_state)
            behavior_probs.append(epsilon_greedy_policy(discrete_state)[action])
            
            if terminated or truncated:
                T = t + 1
            else:
                action = select_action(next_discrete_state, epsilon_greedy_policy)
            
            state = next_state
            discrete_state = next_discrete_state
        
        update_time = t - n_steps + 1
        if update_time >= 0:
            G = 0
            for i in range(update_time, min(update_time + n_steps, T)):
                step_idx = i - update_time
                G += (gamma ** step_idx) * rewards[step_idx]
            
            if update_time + n_steps < T:
                bootstrap_state = states[n_steps]
                G += (gamma ** n_steps) * np.max(Q[bootstrap_state])
            
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
            
            update_state = states.popleft()
            update_action = actions.popleft()
            rewards.popleft()
            behavior_probs.popleft()
            
            Q[update_state][update_action] += alpha * rho * (G - Q[update_state][update_action])
        
        if update_time == T - 1:
            break
        
        t += 1
    
    reward_history.append(episode_reward)
    episode_length_history.append(episode_length)
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Calculate Q-value statistics every 10 episodes
    if episode % 10 == 0:
        q_samples = []
        for _ in range(100):
            random_state = tuple(np.random.randint(0, n_bins) for _ in range(6))
            q_samples.append(np.max(Q[random_state]))
        average_q_values.append(np.mean(q_samples))
        max_q_values.append(np.max(q_samples))
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(reward_history[-100:])
        avg_length = np.mean(episode_length_history[-100:])
        print(f"Episode {episode+1}/{num_episodes}, "
              f"Avg Reward: {avg_reward:.1f}, "
              f"Avg Length: {avg_length:.1f}, "
              f"Epsilon: {epsilon:.3f}")

env.close()

# Plotting results
plt.figure(figsize=(12, 10))

# Episode rewards/lengths
plt.subplot(3, 1, 1)
plt.plot(reward_history, label="Episode Reward", alpha=0.4, color='blue')
window = min(100, len(reward_history))
plt.plot(np.convolve(reward_history, np.ones(window)/window, mode='valid'),
         label=f"{window}-Episode Moving Average", color='red', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"Acrobot Performance with Off-Policy n-Step TD (n={n_steps})")
plt.legend()
plt.grid(alpha=0.3)

# Q-value statistics
plt.subplot(3, 1, 2)
episode_indices = [i*10 for i in range(len(average_q_values))]
plt.plot(episode_indices, average_q_values, label="Average Max Q-Value", color='purple', linewidth=2)
plt.plot(episode_indices, max_q_values, label="Global Max Q-Value", color='green', linewidth=1, alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Q-Value")
plt.title("Q-Value Evolution During Learning")
plt.legend()
plt.grid(alpha=0.3)

# Epsilon decay
epsilon_history = [1.0 * (0.995 ** i) for i in range(num_episodes)]
epsilon_history = [max(0.01, e) for e in epsilon_history]
plt.subplot(3, 1, 3)
plt.plot(epsilon_history, label="Exploration Rate (ε)", color='orange', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration Rate Decay")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Test the learned policy
def test_policy(n_tests=10):
    test_rewards = []
    test_lengths = []
    
    for _ in range(n_tests):
        state, _ = env.reset()
        discrete_state = discretize_state(state)
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = np.argmax(Q[discrete_state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_length += 1
            
            state = next_state
            discrete_state = discretize_state(state)
        
        test_rewards.append(total_reward)
        test_lengths.append(episode_length)
    
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.1f}")
    print(f"Average Episode Length: {np.mean(test_lengths):.1f}")
    print(f"Test Rewards: {test_rewards}")
    print(f"Test Lengths: {test_lengths}")

print("\nTesting the learned policy...")
test_policy()