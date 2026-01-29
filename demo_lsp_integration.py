import asyncio
import os
import sys
import logging
import json
import time

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ide_clients.reference_client import ReferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Demo")

async def run_demo():
    logger.info("--- Starting AlphaEvolve LSP Demo ---")
    
    # 1. Initialize Client
    client = ReferenceClient()
    logger.info("Initializing LSP Client...")
    
    # Add a log file for server debugging
    client.lsp_server_path = 'lsp_server.server'
    
    if await client.initialize():
        logger.info("✅ Client initialized successfully.")
    else:
        logger.error("❌ Failed to initialize client.")
        return

    # 2. Define a sample document with EVOLVE-BLOCK
    document_uri = "file:///demo_script.py"
    document_content = """
import random

# TASK-DEFINITION
# id: sort_optimization
# description: Optimize the sorting algorithm
# function_name_to_evolve: bubble_sort
# input_output_examples:
#   - input: [[3, 1, 2]]
#     output: [1, 2, 3]
# END-TASK-DEFINITION

def main():
    data = [random.randint(0, 100) for _ in range(10)]
    print(f"Original: {data}")
    
    # EVOLVE-BLOCK-START
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    # EVOLVE-BLOCK-END
    
    sorted_data = bubble_sort(data)
    print(f"Sorted: {sorted_data}")

if __name__ == "__main__":
    main()
"""
    
    # 3. Simulate opening the file (send didOpen notification)
    logger.info(f"Opening document: {document_uri}")
    await client._send_notification('textDocument/didOpen', {
        'textDocument': {
            'uri': document_uri,
            'languageId': 'python',
            'version': 1,
            'text': document_content
        }
    })
    
    # 4. Request Code Actions for the EVOLVE-BLOCK
    # The block is roughly lines 15-23
    logger.info("Requesting Code Actions for lines 15-23...")
    actions = await client.request_code_actions(document_uri, 15, 23)
    
    if not actions:
        logger.warning("No actions found.")
        client.shutdown()
        return

    logger.info(f"✅ Received {len(actions)} actions:")
    evolve_action = None
    for action in actions:
        title = action.get('title')
        logger.info(f"  - {title}")
        if "Evolve This Code Block" in title:
            evolve_action = action

    if evolve_action:
        # 5. Execute "Start Evolution"
        command = evolve_action['command']
        cmd_name = command['command']
        cmd_args = command['arguments']
        
        logger.info(f"🚀 Triggering command: {cmd_name}")
        response = await client.execute_command(cmd_name, cmd_args)
        logger.info(f"Server Response: {json.dumps(response, indent=2)}")
        
        if response.get('status') == 'started':
            evolution_id = response['evolution_id']
            block_id = cmd_args[1]
            
            # 6. Monitor Progress - wait for evolution to complete
            logger.info("⏳ Waiting for evolution to complete...")
            max_wait = 10  # seconds
            poll_interval = 0.5
            elapsed = 0
            
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
                progress = await client.execute_command("alphaevolve.showProgress", [document_uri, block_id])
                status = progress.get('status', 'unknown')
                current_progress = progress.get('current_progress', {})
                
                if current_progress.get('message'):
                    logger.info(f"  [{elapsed:.1f}s] {current_progress.get('message')}")
                
                if status == 'complete':
                    logger.info("✅ Evolution completed!")
                    break
                elif status in ['failed', 'stopped']:
                    logger.warning(f"Evolution ended with status: {status}")
                    break
            
            # 7. Show final results
            final_progress = await client.execute_command("alphaevolve.showProgress", [document_uri, block_id])
            logger.info(f"\n📊 Final Results:")
            logger.info(f"  Status: {final_progress.get('status')}")
            
            results = final_progress.get('results', {})
            if results.get('best_code'):
                logger.info(f"  Best Score: {results.get('best_score', 'N/A')}")
                logger.info(f"  Mode: {results.get('mode', 'real')}")
                logger.info(f"\n🧬 Evolved Code:\n{'-'*40}")
                print(results.get('best_code'))
                print('-'*40)

    # 8. Shutdown
    logger.info("Shutting down client...")
    client.shutdown()
    logger.info("--- Demo Completed ---")

if __name__ == "__main__":
    asyncio.run(run_demo())
