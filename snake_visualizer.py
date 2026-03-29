import pygame
import sys

# Colors for our UI
BG_COLOR         = (30, 30, 30)      # Dark gray background
GRID_COLOR       = (50, 50, 50)      # Light gray grid lines
SNAKE_HEAD_COLOR = (29, 158, 117)    # Teal (Matches your Young Agent plot color)
SNAKE_BODY_COLOR = (143, 193, 169)   # Lighter teal
FOOD_COLOR       = (216, 90, 48)     # Coral red (Matches your Old Agent plot color)
TEXT_COLOR       = (255, 255, 255)

def watch_agent(agent, env, cell_size=40, fps=12):
    """
    Runs a single visual episode of the environment using a trained agent.
    """
    pygame.init()
    
    # Calculate screen dimensions based on grid size
    width = env.grid * cell_size
    height = env.grid * cell_size
    
    # Add 40 extra pixels at the top for a scoreboard
    screen = pygame.display.set_mode((width, height + 40))
    pygame.display.set_caption(f"AgeingBipartiteGraph - Age: {agent.age}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20, bold=True)

    state = env.reset()
    done = False
    total_reward = 0
    food_eaten = 0

    while not done:
        # 1. Allow user to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Let the brain decide the next move (using predict, not forward!)
        action = agent.predict(state)
        state, reward, done = env.step(action)
        
        total_reward += reward
        if reward >= 10.0:
            food_eaten += 1

        # 3. Draw everything
        screen.fill(BG_COLOR)

        # Draw Scoreboard background
        pygame.draw.rect(screen, (20, 20, 20), (0, 0, width, 40))
        
        # Draw text
        score_text = font.render(f"Age: {agent.age} | Food: {food_eaten} | Reward: {total_reward:.1f}", True, TEXT_COLOR)
        screen.blit(score_text, (10, 10))

        # Shift everything down by 40 pixels for the game board
        board_surface = pygame.Surface((width, height))
        board_surface.fill(BG_COLOR)

        # Draw grid
        for x in range(0, width, cell_size):
            pygame.draw.line(board_surface, GRID_COLOR, (x, 0), (x, height))
        for y in range(0, height, cell_size):
            pygame.draw.line(board_surface, GRID_COLOR, (0, y), (width, y))

        # Draw food
        fr, fc = env.food
        food_rect = pygame.Rect(fc * cell_size, fr * cell_size, cell_size, cell_size)
        pygame.draw.rect(board_surface, FOOD_COLOR, food_rect, border_radius=5)

        # Draw snake
        for i, (r, c) in enumerate(env.snake):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            if i == 0:
                pygame.draw.rect(board_surface, SNAKE_HEAD_COLOR, rect, border_radius=4)
            else:
                # Shrink the body segments slightly so it looks like a segmented snake
                body_rect = rect.inflate(-4, -4)
                pygame.draw.rect(board_surface, SNAKE_BODY_COLOR, body_rect, border_radius=4)

        # Apply the board surface to the main screen below the scoreboard
        screen.blit(board_surface, (0, 40))

        pygame.display.flip()
        
        # Control the speed of the game
        clock.tick(fps)

    # Pause for 2 seconds when the snake dies/finishes so you can see what happened
    pygame.time.delay(2000)
    pygame.quit()

