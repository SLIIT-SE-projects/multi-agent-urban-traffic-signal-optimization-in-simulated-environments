import os

class Config:
    DEBUG = True
    USE_GUI = True
    STEP_LENGTH = 1.0
    AUTO_START_STEPPING = True
    STEP_DELAY = 0.1
    HOST = '0.0.0.0'
    PORT = 5000

class ProductionConfig(Config):
    DEBUG = False
    USE_GUI = False
    STEP_DELAY = 0.05

class DevelopmentConfig(Config):
    DEBUG = True
    USE_GUI = True

# Load based on FLASK_ENV
env = os.getenv('FLASK_ENV', 'development')
config = ProductionConfig() if env == 'production' else DevelopmentConfig()