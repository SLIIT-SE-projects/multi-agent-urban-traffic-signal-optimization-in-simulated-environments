"""
Domain Entity: TrafficSignal
Represents the phase logic of an intersection.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List

class Phase(BaseModel):
    """
    Represents a single signal phase (e.g., 'GGrr').
    """
    duration: int = Field(..., ge=0, description="Duration in seconds")
    state: str = Field(..., description="SUMO state string (e.g., 'GGrr')")
    min_dur: int = Field(default=5)
    max_dur: int = Field(default=60)

    @field_validator('state')
    @classmethod
    def validate_state_chars(cls, v: str) -> str:
        valid_chars = set("rRgGyYOo")
        if not all(c in valid_chars for c in v):
            raise ValueError(f"Invalid signal state characters: {v}")
        return v

class TrafficLightEntity(BaseModel):
    """
    Represents a physical Traffic Light Controller (TLC).
    """
    id: str
    phases: List[Phase]
    current_phase_index: int = 0

    def get_current_phase(self) -> Phase:
        return self.phases[self.current_phase_index]