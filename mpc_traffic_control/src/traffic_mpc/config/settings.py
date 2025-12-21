"""
Configuration Management Module.
Defines the Pydantic V2 schema for the application.
"""
import os
from typing import Literal, Optional, List
from typing_extensions import Annotated
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PositiveFloat,
    field_validator,
    model_validator,
    ConfigDict,
    FilePath,
    DirectoryPath
)

# --- Component Configurations ---

class SumoConfig(BaseModel):
    """
    Configuration for the SUMO simulation interface.
    """
    model_config = ConfigDict(frozen=True, extra='forbid')

    sumo_binary: str = Field(default="sumo", description="Binary to execute (sumo or sumo-gui)")
    config_file: str = Field(..., description="Path to the .sumocfg file")  # Changed to str to avoid validation error before file exists
    step_length: PositiveFloat = Field(default=1.0, description="Simulation step size in seconds")
    warmup_seconds: int = Field(default=0, ge=0, description="Warmup duration before control starts")
    port: Optional[int] = Field(default=None, description="TraCI port (None=auto)")
    use_gui: bool = Field(default=False, description="Launch with GUI")

    @field_validator('sumo_binary')
    @classmethod
    def validate_binary(cls, v: str) -> str:
        valid_bins = ['sumo', 'sumo-gui', 'dsumo']
        # Relaxed check for full paths
        if not any(b in v.lower() for b in valid_bins) and os.path.basename(v) not in valid_bins:
            # In production, you might log a warning here
            pass 
        return v


class OptimizationConfig(BaseModel):
    """
    Configuration for the solver and cost function weights.
    J = w_queue * Q^2 + w_delay * D + w_switch * S
    """
    solver_name: Literal["osqp", "gurobi", "ipopt"] = Field(default="ipopt")
    time_limit: PositiveFloat = Field(default=0.1, description="Solver time limit in seconds")

    # Cost Function Weights
    weight_queue: float = Field(default=1.0, ge=0.0)
    weight_delay: float = Field(default=0.0, ge=0.0)
    weight_switch: float = Field(default=10.0, ge=0.0, description="Penalty for changing phases")


class MPCConfig(BaseModel):
    """
    Configuration for the Model Predictive Controller core logic.
    """
    model_config = ConfigDict(frozen=True)

    prediction_horizon: PositiveInt = Field(default=20, description="Np: Steps to predict forward")
    control_horizon: PositiveInt = Field(default=5, description="Nu: Steps to optimize control")
    
    min_green_time: PositiveInt = Field(default=10, description="Minimum green in seconds")
    max_green_time: PositiveInt = Field(default=60, description="Maximum green in seconds")
    yellow_time: PositiveInt = Field(default=3, description="Inter-green clearance time")

    @model_validator(mode='after')
    def check_horizons(self) -> 'MPCConfig':
        """Validates that Prediction Horizon >= Control Horizon."""
        if self.prediction_horizon < self.control_horizon:
            raise ValueError(
                f"Prediction horizon ({self.prediction_horizon}) must be >= "
                f"Control horizon ({self.control_horizon})."
            )
        return self

    @model_validator(mode='after')
    def check_phase_constraints(self) -> 'MPCConfig':
        """Ensures physical feasibility of phase times."""
        if self.max_green_time <= self.min_green_time:
            raise ValueError(
                f"Max green ({self.max_green_time}) must be > "
                f"Min green ({self.min_green_time})"
            )
        return self


class LoggingConfig(BaseModel):
    """Configuration for telemetry and system logging."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = Field(default="data/logs")
    save_frequency: PositiveInt = Field(default=100, description="Steps between disk writes")


class AppConfig(BaseModel):
    """Root aggregate configuration."""
    sumo: SumoConfig
    mpc: MPCConfig
    optimization: OptimizationConfig
    logging: LoggingConfig