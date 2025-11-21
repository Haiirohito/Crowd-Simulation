# constants.py
WINDOW_SIZE = 1000
GRID_W = 80
GRID_H = 80
CELL_PIX = WINDOW_SIZE // GRID_W

NUM_AGENTS = 250
AGENT_RADIUS = 0.4

MAX_SPEED = 3.0
SEPARATION_WEIGHT = 6.0
GOAL_WEIGHT = 5.0
DENSITY_WEIGHT = 0.05 # Cost per agent in a cell
SOCIAL_WEIGHT = 15.0 # Force weight for social repulsion
SOCIAL_RADIUS = 1.5 # Radius of social interaction

TIME_STEP = 1.0 / 60.0

MAX_DOORS = 8
