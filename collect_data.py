import numpy as np
import os
from env_vision import SnakeEnv, MAX_STEPS, DELTAS, TURN_R, TURN_L

# ─── Heuristic Policy ────────────────────────────────────────────

def heuristic_policy(env):
    """Deterministic greedy policy for data collection."""
    head = env.snake[0]
    food = env.food
    
    def is_safe(pos):
        r, c = pos
        return 0 <= r < env.grid and 0 <= c < env.grid and pos not in env.snake

    actions = [0, 1, 2]
    best_action = 0
    min_dist = float('inf')
    found_safe_move = False
    
    for action in actions:
        temp_dir = env.direction
        if action == 1: temp_dir = TURN_R[env.direction]
        elif action == 2: temp_dir = TURN_L[env.direction]
        
        dr, dc = DELTAS[temp_dir]
        next_step = (head[0] + dr, head[1] + dc)
        
        if is_safe(next_step):
            found_safe_move = True
            dist = abs(next_step[0] - food[0]) + abs(next_step[1] - food[1])
            if dist < min_dist:
                min_dist = dist
                best_action = action
                
    if not found_safe_move:
        return 0 
        
    return best_action

# ─── Data Collection ─────────────────────────────────────────────

def collect_vision_dataset(target_frames=150000):
    """
    Runs the heuristic agent until the target number of frames is reached.
    Returns a numpy array of shape (N, 1, Grid_Size, Grid_Size).
    """
    env = SnakeEnv()
    observations = []
    episodes = 0
    
    np.random.seed(seed=None)

    print(f"Starting data collection. Target: {target_frames} frames...")
    
    while len(observations) < target_frames:
        env.reset()
        done = False
        episodes += 1
        
        while not done and len(observations) < target_frames:
            # 1. Grab the visual grid (Ensure your SnakeEnv._state() returns the 2D matrix)
            current_grid = env._state() 
            observations.append(current_grid)
            
            # 2. Step the environment
            action = heuristic_policy(env)
            _, _, done = env.step(action)
            
            # Break infinite loops if the heuristic gets stuck
            if env.steps > MAX_STEPS:
                break
                
        # Print progress every 100 episodes
        if episodes % 100 == 0:
            print(f"Collected {len(observations)} / {target_frames} frames...")

    # Convert to PyTorch format: (Batch, Channels, Height, Width)
    # Adding the channel dimension (1 for grayscale-like grid)
    dataset = np.array(observations, dtype=np.float32)
    dataset = dataset[:, np.newaxis, :, :] 
    
    print(f"Collection complete! Total episodes run: {episodes}")
    return dataset

# ─── Execution & Saving ──────────────────────────────────────────

def main():
    # Adjust this number depending on how much RAM/Disk space you want to use.
    TOTAL_FRAMES = 200000
    FILENAME = "snake_vision_data.npy"
    
    # 1. Collect
    dataset = collect_vision_dataset(target_frames=TOTAL_FRAMES)
    
    # 2. Verify Shape
    print(f"Dataset Shape: {dataset.shape}") # Expected: (20000, 1, 10, 10)
    
    # 3. Save to disk
    save_path = os.path.join(os.getcwd(), FILENAME)
    np.save(save_path, dataset)
    print(f"Successfully saved dataset to: {save_path}")
    print(f"File size: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
