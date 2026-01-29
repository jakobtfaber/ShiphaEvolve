import pytest
import asyncio
from lsprotocol.types import *
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lsp_server.alphaevolve_lsp_bridge import AlphaEvolveLSPBridge

@pytest.fixture
def bridge():
    return AlphaEvolveLSPBridge()

@pytest.mark.asyncio
async def test_initialize(bridge):
    """Test LSP initialization."""
    result = bridge.m_initialize()
    # Result is now a dict (serialized from InitializeResult)
    assert 'capabilities' in result
    assert 'codeActionProvider' in result['capabilities']

@pytest.mark.asyncio
async def test_evolve_block_detection(bridge):
    """Test detection of EVOLVE-BLOCK markers."""
    document = """
def main():
    # EVOLVE-BLOCK-START
    def sort_array(arr):
        for i in range(len(arr)):
            for j in range(len(arr)-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    # EVOLVE-BLOCK-END
"""
    
    blocks = bridge._find_evolve_blocks(
        document,
        Range(start=Position(line=0, character=0), end=Position(line=10, character=0))
    )
    
    assert len(blocks) == 1
    assert any(b.startswith('evolve_block_') for b in blocks.keys())

@pytest.mark.asyncio
async def test_task_extraction(bridge):
    """Test task definition extraction."""
    code = """
# TASK-DEFINITION
# id: sort
# description: Sort an array
# function_name_to_evolve: sort_array
# END-TASK-DEFINITION

def sort_array(arr):
    pass
"""
    
    task_def = bridge._extract_task_definition(code)
    assert task_def['id'] == 'sort'
    assert task_def['function_name_to_evolve'] == 'sort_array'

@pytest.mark.asyncio
async def test_code_actions(bridge):
    """Test code action provision."""
    bridge._document_cache = {
        'file:///test.py': "# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END"
    }
    
    # Use dictionaries instead of lsprotocol types (as pyls_jsonrpc passes dicts)
    actions = bridge.m_text_document__code_action(
        textDocument={'uri': 'file:///test.py'},
        range={'start': {'line': 0, 'character': 0}, 'end': {'line': 3, 'character': 0}},
        context={'diagnostics': []}
    )
    
    # Actions is now a list of dicts
    assert len(actions) > 0
    assert any(a.get('title') == "🚀 Evolve This Code Block" for a in actions)
