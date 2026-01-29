from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, List

class IDEClientBase(ABC):
    """
    Abstract base class for IDE clients.
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        pass
    
    @abstractmethod
    async def find_evolve_blocks(self, document_path: str) -> Dict:
        pass
    
    @abstractmethod
    async def request_code_actions(self, document_uri: str, line_start: int, line_end: int) -> list:
        pass
    
    @abstractmethod
    async def execute_command(self, command: str, arguments: list) -> Dict:
        pass
    
    @abstractmethod
    async def register_command_handler(self, command: str, handler: Callable) -> None:
        pass
    
    @abstractmethod
    async def show_status_message(self, message: str, duration_ms: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    async def show_input_dialog(self, prompt: str, default_value: str = "") -> Optional[str]:
        pass
    
    @abstractmethod
    async def show_diff_view(self, original: str, evolved: str, original_name: str = "Original", evolved_name: str = "Evolved") -> bool:
        pass
    
    @abstractmethod
    async def on_evolution_progress(self, evolution_id: str, progress: Dict) -> None:
        pass
    
    @abstractmethod
    async def on_evolution_complete(self, evolution_id: str, result: Dict) -> None:
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        pass
