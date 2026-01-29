import pytest
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ide_clients.reference_client import ReferenceClient
from ide_clients.generic_lsp_client import GenericLSPClient

def test_client_instantiation():
    client = ReferenceClient()
    assert isinstance(client, GenericLSPClient)
    assert client.ide_name == 'reference_client'
