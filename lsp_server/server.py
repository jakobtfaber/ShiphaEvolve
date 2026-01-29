import argparse
import logging
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyls_jsonrpc.endpoint import Endpoint
from pyls_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter
from lsp_server.alphaevolve_lsp_bridge import AlphaEvolveLSPBridge

def setup_logging(log_file: str = None):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

def start_lsp_server(port: int = None):
    logger = logging.getLogger(__name__)
    logger.info("Starting AlphaEvolve LSP Server")
    
    bridge = AlphaEvolveLSPBridge()
    
    if port:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('localhost', port))
        sock.listen(1)
        logger.info(f"Listening on port {port}")
        client_sock, _ = sock.accept()
        reader = JsonRpcStreamReader(client_sock.makefile('rb'))
        writer = JsonRpcStreamWriter(client_sock.makefile('wb'))
    else:
        # Use standard streams if stdio
        # Note: JsonRpcStreamReader expects a binary stream
        reader = JsonRpcStreamReader(sys.stdin.buffer)
        writer = JsonRpcStreamWriter(sys.stdout.buffer)
    
    endpoint = Endpoint(bridge, writer.write)
    logger.info("LSP Server initialized. Waiting for connections...")
    
    # In stdio mode, we might need to be careful about not blocking everything if listen() blocks.
    # But for stdio LSP, blocking on stdin is usually correct.
    reader.listen(endpoint.consume)

def main():
    parser = argparse.ArgumentParser(description='AlphaEvolve LSP Server')
    parser.add_argument('--port', type=int, help='Port to listen on (if not provided, uses stdio)')
    parser.add_argument('--log-file', help='Log file path')
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    try:
        start_lsp_server(args.port)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.exception(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
