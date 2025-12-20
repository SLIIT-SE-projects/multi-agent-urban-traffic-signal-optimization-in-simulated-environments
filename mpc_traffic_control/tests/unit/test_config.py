import pytest
from pydantic import ValidationError
from src.traffic_mpc.config.settings import MPCConfig, SumoConfig

def test_mpc_valid_config():
    """Test that a valid configuration is accepted."""
    cfg = MPCConfig(
        prediction_horizon=20,
        control_horizon=5,
        min_green_time=10,
        max_green_time=60
    )
    assert cfg.prediction_horizon == 20

def test_mpc_horizon_error():
    """Test that Np < Nu raises a validation error."""
    with pytest.raises(ValidationError) as excinfo:
        MPCConfig(
            prediction_horizon=5,  # Too short!
            control_horizon=10,
            min_green_time=10,
            max_green_time=60
        )
    # Check that the error message contains our custom logic
    assert "Prediction horizon" in str(excinfo.value)

def test_mpc_green_time_error():
    """Test that max_green <= min_green raises error."""
    with pytest.raises(ValidationError):
        MPCConfig(
            prediction_horizon=20,
            control_horizon=5,
            min_green_time=30,
            max_green_time=15 # Impossible!
        )

def test_sumo_binary_validation():
    """Test the fuzzy matching for sumo binary names."""
    cfg = SumoConfig(
        sumo_binary="C:/Program Files/Eclipse/Sumo/bin/sumo-gui.exe",
        config_file="dummy.sumocfg"
    )
    assert cfg.sumo_binary == "C:/Program Files/Eclipse/Sumo/bin/sumo-gui.exe"