import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

class SimConfig:
    SUMO_BINARY = "sumo-gui"
    SIMULATION_FOLDER = os.path.join(PROJECT_ROOT, "simulation")
    NET_FILE = os.path.join(SIMULATION_FOLDER, "network.net.xml")
    ROUTE_FILE = os.path.join(SIMULATION_FOLDER, "routes.rou.xml")
    SUMO_CFG = os.path.join(SIMULATION_FOLDER, "scenario.sumocfg")
    
    STEP_LENGTH = 1.0  # Step size
    WARMUP_STEPS = 500 # Seconds to run before controlling
    MIN_GREEN_TIME = 5 # Safety constraint

class GraphConfig:
    NUM_SIGNAL_PHASES = 3  # Fixed to 3 based on your map
    INTERSECTION_INPUT_DIM = NUM_SIGNAL_PHASES + 1  # Phases + Time feature
    LANE_INPUT_DIM = 2     # Queue + Speed (Add more if using CO2 etc)

class ModelConfig:
    HIDDEN_DIM = 32
    NUM_HEADS = 2      # For GATConv
    DROPOUT_RATE = 0.3 # Uncertainty Mechanism
    USE_GRU = True     # Recurrent Wrapper

class TrainConfig:
    # DATA CONFIGURATION
    STEPS_TO_COLLECT = 3600 * 2  # 2 hours of simulation data
    # SSL TRAINING SETTINGS
    SSL_EPOCHS = 10
    SSL_LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8

    # MARL TRAINING SETTINGS
    MARL_EPISODES = 50
    MARL_STEPS_PER_EPISODE = 1000
    MARL_LEARNING_RATE = 0.0005
    ACTION_INTERVAL = 5    # Action every 5 seconds

    # Epsilon Greedy (Exploration)
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.90

    # Reward Weights
    W_QUEUE = 2.0
    W_WAIT = 0.001

    # INFERENCE & SAFETY
    UNCERTAINTY_THRESHOLD = 0.05
    MC_SAMPLES = 30

class FileConfig:
    EXPERIMENTS_FOLDER = os.path.join(PROJECT_ROOT, "experiments")
    RAW_DATA_DIR = os.path.join(EXPERIMENTS_FOLDER, "raw_data")
    DATASET_PATH = os.path.join(RAW_DATA_DIR, "traffic_data_1hr.pt")

    MODELS_DIR = os.path.join(EXPERIMENTS_FOLDER, "saved_models")
    PRETRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "pretrained_gnn.pth")
    FINAL_MARL_MODEL_PATH = os.path.join(MODELS_DIR, "final_marl_model.pth")

    PLOTS_DIR = os.path.join(EXPERIMENTS_FOLDER, "plots")