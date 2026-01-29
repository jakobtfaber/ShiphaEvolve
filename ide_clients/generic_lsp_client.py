import asyncio
import json
from typing import Dict, Optional, Callable, List
from .base_client import IDEClientBase

class GenericLSPClient(IDEClientBase):
    """
    Generic LSP client implementation.
    """
    
    def __init__(self, lsp_server_path: str, ide_name: str = "generic", config: Dict = None):
        self.lsp_server_path = lsp_server_path
        self.ide_name = ide_name
        self.config = config or {}
        
        self.process = None
        self.reader = None
        self.writer = None
        self.message_id = 0
        self.pending_requests = {}
    
    async def initialize(self) -> bool:
        try:
            self.process = await asyncio.create_subprocess_exec(
                'python', '-m', self.lsp_server_path,
                stdout=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.reader = asyncio.StreamReader()
            # We read from process stdout. But asyncio.create_subprocess_exec gives us a reader in stdout.
            # self.reader = self.process.stdout # This is already a StreamReader
            # But the roadmap code does:
            # self.reader = asyncio.StreamReader()
            # self.writer = self.process.stdin
            # This looks wrong if self.process.stdout is used directly.
            # Correct approach:
            self.reader = self.process.stdout
            self.writer = self.process.stdin
            
            asyncio.create_task(self._listen_for_messages())
            
            init_result = await self._send_request('initialize', {
                'processId': None,
                'rootPath': None,
                'capabilities': {},
            })
            await self._send_notification('initialized', {})
            return init_result is not None
        except Exception as e:
            print(f"Failed to initialize LSP: {e}")
            return False
    
    async def find_evolve_blocks(self, document_path: str) -> Dict:
        return {}
    
    async def request_code_actions(self, document_uri: str, line_start: int, line_end: int) -> list:
        result = await self._send_request('textDocument/codeAction', {
            'textDocument': {'uri': document_uri},
            'range': {
                'start': {'line': line_start, 'character': 0},
                'end': {'line': line_end, 'character': 0}
            },
            'context': {'diagnostics': []}
        })
        return result or []
    
    async def execute_command(self, command: str, arguments: list) -> Dict:
        result = await self._send_request('workspace/executeCommand', {
            'command': command,
            'arguments': arguments
        })
        return result or {}
    
    async def register_command_handler(self, command: str, handler: Callable) -> None:
        pass
    
    async def show_status_message(self, message: str, duration_ms: Optional[int] = None) -> None:
        pass
    
    async def show_input_dialog(self, prompt: str, default_value: str = "") -> Optional[str]:
        pass
    
    async def show_diff_view(self, original: str, evolved: str, original_name: str = "Original", evolved_name: str = "Evolved") -> bool:
        pass
    
    async def on_evolution_progress(self, evolution_id: str, progress: Dict) -> None:
        pass
    
    async def on_evolution_complete(self, evolution_id: str, result: Dict) -> None:
        pass
    
    # LSP Protocol Implementation
    
    async def _send_request(self, method: str, params: Dict) -> Optional[Dict]:
        message_id = self.message_id
        self.message_id += 1
        request = {
            'jsonrpc': '2.0',
            'id': message_id,
            'method': method,
            'params': params
        }
        self.pending_requests[message_id] = asyncio.Future()
        
        message_json = json.dumps(request)
        self.writer.write(f"Content-Length: {len(message_json)}\r\n\r\n{message_json}".encode())
        await self.writer.drain()
        
        try:
            response = await asyncio.wait_for(self.pending_requests[message_id], timeout=30.0)
            return response
        except asyncio.TimeoutError:
            print(f"Request timeout: {method}")
            return None
        finally:
            del self.pending_requests[message_id]
    
    async def _send_notification(self, method: str, params: Dict) -> None:
        notification = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params
        }
        message_json = json.dumps(notification)
        self.writer.write(f"Content-Length: {len(message_json)}\r\n\r\n{message_json}".encode())
        await self.writer.drain()
    
    async def _listen_for_messages(self):
        # Basic LSP header parsing
        while True:
            try:
                line = await self.reader.readuntil(b'\r\n')
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                
                if line.startswith("Content-Length:"):
                    length = int(line.split(":")[1].strip())
                    # Skip remaining headers until empty line
                    while True:
                        header_line = await self.reader.readuntil(b'\r\n')
                        if header_line == b'\r\n' or header_line.strip() == b'':
                            break
                    content = await self.reader.readexactly(length)
                    message = json.loads(content.decode('utf-8'))
                    await self._handle_message(message)
            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                print(f"Message listening error: {e}")
                break
    
    async def _handle_message(self, message: Dict):
        if 'id' in message and 'method' not in message: # Response
            message_id = message['id']
            if message_id in self.pending_requests:
                if 'error' in message:
                    self.pending_requests[message_id].set_exception(Exception(message['error']))
                else:
                    self.pending_requests[message_id].set_result(message.get('result'))
        elif 'method' in message: # Notification/Request
            await self._handle_server_request(message['method'], message.get('params', {}))
    
    async def _handle_server_request(self, method: str, params: Dict):
        if method == 'window/showMessage':
            await self.show_status_message(params.get('message'))
        elif method == 'alphaevolve/progress':
            await self.on_evolution_progress(params.get('evolution_id'), params.get('progress', {}))
        elif method == 'alphaevolve/complete':
            await self.on_evolution_complete(params.get('evolution_id'), params.get('result', {}))
            
    def shutdown(self) -> None:
        if self.process:
            self.process.terminate()
