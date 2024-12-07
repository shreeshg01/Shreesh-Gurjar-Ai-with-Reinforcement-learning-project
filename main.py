import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Define the GridWorld Environment with Tkinter visualization
class GridWorldGUI:
    def __init__(self, size=10, cell_size=50, dynamic_wall=None, move_interval=10, wall_size=3, move_delay=0.05):
        self.size = size
        self.cell_size = cell_size
        self.window = tk.Tk()
        self.window.title("GridWorld Environment")
        self.canvas = tk.Canvas(self.window, width=self.size * self.cell_size, height=self.size * self.cell_size)
        self.canvas.pack()
        self.maze = self.create_maze()
        self.reset()
        
        # Dynamic wall parameters
        self.dynamic_wall = dynamic_wall  # Boolean flag to enable dynamic wall
        self.move_interval = move_interval  # Move every 'move_interval' steps
        self.steps_since_move = 0
        self.wall_direction = 1  # 1 for down, -1 for up
        self.wall_size = wall_size  # Number of rows the wall spans
        self.move_delay = move_delay  # Delay in seconds between steps

        if self.dynamic_wall:
            self.initialize_dynamic_wall()

    def create_maze(self):
        maze = np.ones((self.size, self.size), dtype=int)
        maze[1:4, 3] = 0  # Vertical obstacle (potential dynamic wall)
        maze[6, 1:5] = 0  # Horizontal obstacle
        maze[4:8, 7] = 0  # Another vertical obstacle
        return maze

    def initialize_dynamic_wall(self):
        # Define the dynamic wall's initial position and fixed movement range
        self.initial_wall_position = 1  # Starting top row index for the dynamic wall
        self.dynamic_wall_position = self.initial_wall_position  # Current top row index
        self.dynamic_wall_range = (self.initial_wall_position, self.initial_wall_position + 2)  # Oscillate between initial and initial +2
        self.dynamic_wall_col = 3  # Column of the dynamic wall
        self.wall_size = 3  # Number of rows the wall spans

        # Set the initial position of the dynamic wall
        for row in range(self.dynamic_wall_position, self.dynamic_wall_position + self.wall_size):
            self.maze[row, self.dynamic_wall_col] = 0

    def move_dynamic_wall(self):
        if not self.dynamic_wall:
            return

        # Erase current dynamic wall
        for row in range(self.dynamic_wall_position, self.dynamic_wall_position + self.wall_size):
            self.maze[row, self.dynamic_wall_col] = 1  # Set to free space

        # Update the dynamic wall's position by moving two rows at a time
        self.dynamic_wall_position += self.wall_direction * 2

        # Check and reverse direction if limits are reached
        if self.dynamic_wall_position > self.dynamic_wall_range[1]:
            self.dynamic_wall_position = self.dynamic_wall_range[1]
            self.wall_direction = -1  # Reverse direction to move up
        elif self.dynamic_wall_position < self.dynamic_wall_range[0]:
            self.dynamic_wall_position = self.dynamic_wall_range[0]
            self.wall_direction = 1  # Reverse direction to move down

        # Ensure the wall stays within bounds
        self.dynamic_wall_position = max(self.dynamic_wall_range[0], min(self.dynamic_wall_position, self.dynamic_wall_range[1]))

        # Set the new position of the dynamic wall
        for row in range(self.dynamic_wall_position, self.dynamic_wall_position + self.wall_size):
            if row < self.size:  # Prevent index out of bounds
                self.maze[row, self.dynamic_wall_col] = 0

        print(f"Dynamic wall moved to rows {self.dynamic_wall_position} to {self.dynamic_wall_position + self.wall_size -1}, column {self.dynamic_wall_col}")

    def reset_dynamic_wall_move_counter(self):
        self.steps_since_move = 0

    def reset(self):
        self.state = (0, 0)  # Start at top-left corner
        self.draw_grid()
        self.draw_agent(self.state, "blue")

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.size):
            for j in range(self.size):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                color = "black" if self.maze[i, j] == 0 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        # Goal state
        goal_x1, goal_y1 = (self.size - 1) * self.cell_size, (self.size - 1) * self.cell_size
        goal_x2, goal_y2 = goal_x1 + self.cell_size, goal_y1 + self.cell_size
        self.canvas.create_rectangle(goal_x1, goal_y1, goal_x2, goal_y2, fill="green", outline="black")
        self.window.update()

    def draw_agent(self, state, color):
        i, j = state
        x1, y1 = j * self.cell_size, i * self.cell_size
        x2, y2 = x1 + self.cell_size, y1 + self.cell_size
        # Draw a smaller oval to represent the agent within the cell
        padding = self.cell_size * 0.2
        self.canvas.create_oval(x1 + padding, y1 + padding, x2 - padding, y2 - padding, fill=color, outline="black")
        self.window.update()

    def step(self, action):
        i, j = self.state
        if action == 0 and i > 0 and self.maze[i-1, j] == 1:  # Up
            i -= 1
        elif action == 1 and i < self.size - 1 and self.maze[i+1, j] == 1:  # Down
            i += 1
        elif action == 2 and j > 0 and self.maze[i, j-1] == 1:  # Left
            j -= 1
        elif action == 3 and j < self.size - 1 and self.maze[i, j+1] == 1:  # Right
            j += 1
        self.state = (i, j)
        self.draw_grid()
        return self.state, self.reward(), self.done()

    def reward(self):
        return 1 if self.state == (self.size - 1, self.size - 1) else -0.1

    def done(self):
        return self.state == (self.size - 1, self.size - 1)

    def update_wall_movement(self):
        if not self.dynamic_wall:
            return
        self.steps_since_move += 1
        if self.steps_since_move >= self.move_interval:
            self.move_dynamic_wall()
            self.draw_grid()
            self.reset_dynamic_wall_move_counter()

        # Introduce delay to slightly slow down the movement and agent steps
        time.sleep(self.move_delay)

# Q-Learning Algorithm
def q_learning_gui(env, episodes, alpha, gamma, epsilon):
    Q = {}
    for state in [(i, j) for i in range(env.size) for j in range(env.size)]:
        Q[state] = [0] * 4

    rewards = []

    for episode in range(episodes):
        state = (0, 0)
        total_reward = 0

        print(f"\nQ-Learning Episode {episode + 1}:")
        while True:
            if np.random.rand() < epsilon:
                action = random.choice([0, 1, 2, 3])  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done = env.step(action)
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            # Terminal output for each step
            print(f"Step: State={state}, Action={action}, Reward={reward}, Next State={next_state}")

            state = next_state
            total_reward += reward

            env.draw_agent(state, "blue")  # Q-learning agent is blue

            # Update dynamic wall if applicable
            env.update_wall_movement()

            if done:
                print(f"Q-Learning reached goal with total reward: {total_reward}")
                break
        rewards.append(total_reward)

    return Q, rewards

# Dyna-Q Algorithm
def dyna_q_gui(env, episodes, alpha, gamma, epsilon, planning_steps):
    Q = {}
    model = {}
    for state in [(i, j) for i in range(env.size) for j in range(env.size)]:
        Q[state] = [0] * 4

    rewards = []

    for episode in range(episodes):
        state = (0, 0)
        total_reward = 0

        print(f"\nDyna-Q Episode {episode + 1}:")
        while True:
            if np.random.rand() < epsilon:
                action = random.choice([0, 1, 2, 3])  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done = env.step(action)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # Terminal output for each step
            print(f"Step: State={state}, Action={action}, Reward={reward}, Next State={next_state}")

            model[(state, action)] = (next_state, reward)

            # Planning step (simulated experiences)
            for _ in range(planning_steps):
                sim_state, sim_action = random.choice(list(model.keys()))
                sim_next_state, sim_reward = model[(sim_state, sim_action)]
                Q[sim_state][sim_action] += alpha * (sim_reward + gamma * np.max(Q[sim_next_state]) - Q[sim_state][sim_action])

            state = next_state
            total_reward += reward

            env.draw_agent(state, "red")  # Dyna-Q agent is red

            # Update dynamic wall if applicable
            env.update_wall_movement()

            if done:
                print(f"Dyna-Q reached goal with total reward: {total_reward}")
                break
        rewards.append(total_reward)

    return Q, rewards

# Plot the results for comparison
def plot_results(q_rewards, dyna_q_rewards):
    # Calculate average reward for each episode
    avg_q_rewards = np.cumsum(q_rewards) / np.arange(1, len(q_rewards) + 1)
    avg_dyna_q_rewards = np.cumsum(dyna_q_rewards) / np.arange(1, len(dyna_q_rewards) + 1)

    plt.figure(figsize=(18, 6))

    # Plot the Q-learning average rewards
    plt.subplot(1, 3, 1)  # Create the first subplot
    plt.plot(avg_q_rewards, label="Q-Learning (Avg)", color="blue", linestyle='--')
    plt.xlabel("Episodes")
    plt.ylabel("Average Total Reward")
    plt.title("Q-Learning Performance (Avg Reward)")
    plt.legend()
    plt.grid(True)

    # Plot the Dyna-Q average rewards
    plt.subplot(1, 3, 2)  # Create the second subplot
    plt.plot(avg_dyna_q_rewards, label="Dyna-Q (Avg)", color="red", linestyle='-')
    plt.xlabel("Episodes")
    plt.ylabel("Average Total Reward")
    plt.title("Dyna-Q Performance (Avg Reward)")
    plt.legend()
    plt.grid(True)

    # Plot the total rewards for both Q-learning and Dyna-Q
    plt.subplot(1, 3, 3)  # Create the third subplot
    plt.plot(np.cumsum(q_rewards), label="Q-Learning (Total)", color="blue", linestyle='--')
    plt.plot(np.cumsum(dyna_q_rewards), label="Dyna-Q (Total)", color="red", linestyle='-')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust layout for better spacing between subplots
    plt.show()

def run_experiment():
    num_runs = 2  # Number of independent runs for averaging
    episodes = 100
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    planning_steps = 5

    # Initialize accumulators for rewards
    q_rewards_accum = np.zeros(episodes)
    dyna_q_rewards_accum = np.zeros(episodes)

    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")

        # Initialize the environment with a dynamic wall
        env = GridWorldGUI(
            size=10,
            cell_size=50,
            dynamic_wall=True,
            move_interval=10,  # Wall moves every 10 agent steps
            wall_size=3,
            move_delay=0.05  # Delay of 50ms between steps
        )
        env.reset()

        # Run Q-learning
        _, q_rewards = q_learning_gui(env, episodes, alpha, gamma, epsilon)
        q_rewards_accum += np.array(q_rewards)

        # Reset environment for Dyna-Q
        env.reset()

        # Run Dyna-Q
        _, dyna_q_rewards = dyna_q_gui(env, episodes, alpha, gamma, epsilon, planning_steps)
        dyna_q_rewards_accum += np.array(dyna_q_rewards)

        # Close the environment's Tkinter window after each run to prevent multiple windows
        env.window.destroy()

    # Calculate average rewards across all runs
    avg_q_rewards = q_rewards_accum / num_runs
    avg_dyna_q_rewards = dyna_q_rewards_accum / num_runs

    # Plot results
    plot_results(avg_q_rewards, avg_dyna_q_rewards)

    # Optionally, keep the Tkinter window open (only the last one, if any)
    # env.window.mainloop()

# Start experiment
if __name__ == "__main__":
    run_experiment()
