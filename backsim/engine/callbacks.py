from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backsim.engine.engine import SimulationEngine


class SimulationEngineCallback(ABC):
    """
    Base callback interface for the simulation engine.
    Callbacks can be used for logging, monitoring, or any side effects.
    """

    @abstractmethod
    def on_simulation_start(self, engine: "SimulationEngine"):
        pass

    @abstractmethod
    def on_step_start(self, engine: "SimulationEngine", timestamp: datetime):
        pass

    @abstractmethod
    def on_step_end(self, engine: "SimulationEngine", timestamp: datetime):
        pass

    @abstractmethod
    def on_simulation_end(self, engine: "SimulationEngine"):
        pass
