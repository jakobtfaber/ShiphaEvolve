from .generic_lsp_client import GenericLSPClient
from .base_client import IDEClientBase

class ReferenceClient(GenericLSPClient):
    """Reference implementation of an LSP client."""
    
    def __init__(self):
        super().__init__(
            lsp_server_path='lsp_server.server',
            ide_name='reference_client'
        )
    
    async def show_status_message(self, message: str, duration_ms=None):
        print(f"[Client Status] {message}")
    
    async def show_diff_view(self, original, evolved, **kwargs):
        # Reference implementation for diff display
        return True
