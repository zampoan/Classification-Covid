import pygame
import time
import random

# Initialize Pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set the width and height of each grid cell
GRID_SIZE = 20
GRID_WIDTH = 40
GRID_HEIGHT = 30

# Set the width and height of the game window
WINDOW_WIDTH = GRID_SIZE * GRID_WIDTH
WINDOW_HEIGHT = GRID_SIZE * GRID_HEIGHT

# Set the game speed (lower value means faster speed)
SNAKE_SPEED = 15

# Set the direction of the snake
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Create the game window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game")

# Initialize clock
clock = pygame.time.Clock()

# Function to display messages on the screen
def display_message(message, color, position):
    font_style = pygame.font.SysFont(None, 30)
    rendered_message = font_style.render(message, True, color)
    window.blit(rendered_message, position)

# Function to draw the snake
def draw_snake(snake_body):
    for body_part in snake_body:
        pygame.draw.rect(window, GREEN, [body_part[0], body_part[1], GRID_SIZE, GRID_SIZE])

# Function to run the snake game
def run_game():
    # Initialize game variables
    game_over = False
    game_quit = False

    # Initialize the snake's position and direction
    snake_x = WINDOW_WIDTH // 2
    snake_y = WINDOW_HEIGHT // 2
    snake_direction = RIGHT

    # Initialize the snake's body
    snake_body = []
    snake_length = 1

    # Generate random positions for the food
    food_x = round(random.randrange(0, WINDOW_WIDTH - GRID_SIZE) / GRID_SIZE) * GRID_SIZE
    food_y = round(random.randrange(0, WINDOW_HEIGHT - GRID_SIZE) / GRID_SIZE) * GRID_SIZE

    # Game loop
    while not game_quit:
        while game_over:
            window.fill(BLACK)
            display_message("Game Over! Press Q-Quit or C-Play Again", RED, (WINDOW_WIDTH / 6, WINDOW_HEIGHT / 3))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_quit = True
                    game_over = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_quit = True
                        game_over = False
                    if event.key == pygame.K_c:
                        run_game()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_quit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake_direction != DOWN:
                    snake_direction = UP
                if event.key == pygame.K_DOWN and snake_direction != UP:
                    snake_direction = DOWN
                if event.key == pygame.K_LEFT and snake_direction != RIGHT:
                    snake_direction = LEFT
                if event.key == pygame.K_RIGHT and snake_direction != LEFT:
                    snake_direction = RIGHT

        # Update snake's position
        if snake_direction == UP:
            snake_y -= GRID_SIZE
        if snake_direction == DOWN:
            snake_y += GRID_SIZE
        if snake_direction == LEFT:
            snake_x -= GRID_SIZE
        if snake_direction == RIGHT:
            snake_x += GRID_SIZE

        # Check for collision with the boundary
        if snake_x < 0 or snake_x >= WINDOW_WIDTH or snake_y < 0 or snake_y >= WINDOW_HEIGHT:
            game_over = True

        # Check for collision with the snake's body
        for body_part in snake_body[1:]:
            if body_part[0] == snake_x and body_part[1] == snake_y:
                game_over = True

        # Update snake's body
        snake_head = []
        snake_head.append(snake_x)
        snake_head.append(snake_y)
        snake_body.append(snake_head)
        if len(snake_body) > snake_length:
            del snake_body[0]

        # Check for collision with the food
        if snake_x == food_x and snake_y == food_y:
            food_x = round(random.randrange(0, WINDOW_WIDTH - GRID_SIZE) / GRID_SIZE) * GRID_SIZE
            food_y = round(random.randrange(0, WINDOW_HEIGHT - GRID_SIZE) / GRID_SIZE) * GRID_SIZE
            snake_length += 1

        # Update the game window
        window.fill(BLACK)
        pygame.draw.rect(window, WHITE, [food_x, food_y, GRID_SIZE, GRID_SIZE])
        draw_snake(snake_body)
        pygame.display.update()

        # Set the game speed
        clock.tick(SNAKE_SPEED)

    # Quit Pygame
    pygame.quit()

# Run the game
run_game()
