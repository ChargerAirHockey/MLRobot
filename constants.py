# Define boundaries of table
X_MAX = 100
X_MIN = -100
Y_MAX = 100
Y_MIN = 0

# Define constants for model
NUM_ENV_VAR = 5     # X, Y, X_SPEED, Y_SPEED, ROBOT_POS
NUM_ACTIONS = 11   # MOVE_UP(1-5), MOVE_DOWN(1-5), HIT_PUCK(L/R), DO NOTHING

# Define Hyper-parameters
ALPHA = 0                   # Learning rate
GAMMA = 0                   # Value between 0 and 1 used for importance of future rewards (static or dynamic)
LAMBDA = .00001                  # Decay rate for epsilon
MAX_EPS = 0.9               # Value that epsilon starts at
MIN_EPS = 0.03              # Value that epsilon will decay to
