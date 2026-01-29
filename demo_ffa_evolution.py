#!/usr/bin/env python3
"""
FFA (Fast Folding Algorithm) Evolution Demo

Demonstrates AlphaEvolve's LLM-based evolutionary optimization on a
real scientific computing problem: optimizing the boxcar matched filter
used in pulsar periodicity searches.

Usage:
    python demo_ffa_evolution.py [--enable-hints]

Requirements:
    One of the following environment variables must be set:
    - OPENAI_API_KEY (for OpenAI models)
    - GEMINI_API_KEY (for Google Gemini models)
    - FLASH_API_KEY  (for Flash/custom models)
"""

import asyncio
import os
import sys
import logging
import json
import time
import argparse

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =============================================================================
# API KEY CHECK
# =============================================================================

API_KEY_ERROR_MESSAGE = """
Error: No LLM API key configured.

Set one of the following environment variables to run evolution:
  - OPENAI_API_KEY (for OpenAI models)
  - GEMINI_API_KEY (for Google Gemini models)
  - FLASH_API_KEY  (for Flash/custom models)
"""

def check_api_keys() -> str | None:
    """Check for available API keys. Returns the key name if found, None otherwise."""
    for key_name in ["OPENAI_API_KEY", "GEMINI_API_KEY", "FLASH_API_KEY"]:
        if os.environ.get(key_name):
            return key_name
    return None


# =============================================================================
# DEMO IMPLEMENTATION
# =============================================================================

from ide_clients.reference_client import ReferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FFA-Demo")


def get_ffa_document_content(enable_hints: bool = False) -> str:
    """Generate the FFA demo document with EVOLVE-BLOCK."""
    
    # Hint text (included only if --enable-hints is passed)
    hint_section = ""
    if enable_hints:
        hint_section = """
    # HINTS:
    # - Consider using cumulative sums (prefix sums) for O(n) complexity
    # - NumPy has efficient vectorized operations that avoid Python loops
    # - Prefix-sum: result[i] = prefix[i + width] - prefix[i]
    # - Or use np.convolve with extended array for circular convolution
"""
    
    return f'''#!/usr/bin/env python3
"""
Boxcar Matched Filter for Fast Folding Algorithm (FFA)

This is the computational bottleneck in pulsar periodicity searches.
Goal: Optimize from O(n * width) to O(n) while maintaining accuracy.
"""

import numpy as np

# TASK-DEFINITION
# id: ffa_boxcar_filter_v1
# description: Optimize the boxcar_matched_filter function. It must use CIRCULAR convolution (wrap-around at array boundaries). Input is numpy array (profile) and int (width). Output is array of SAME LENGTH as input. The function computes a sliding sum of 'width' consecutive elements with wrap-around, then divides by sqrt(width) for normalization.
# function_name_to_evolve: boxcar_matched_filter
# allowed_imports: [numpy, math]
# input_output_examples:
#   - input: [[1.0, 2.0, 3.0, 4.0], 2]
#     output: [2.12132, 3.53553, 4.94975, 3.53553]
# constraints:
#   - correctness_tolerance: 1e-8
#   - snr_error_tolerance: 0.01
#   - runtime_constraint: within 5% of baseline
# expert_knowledge: |
#   WORKING BASELINE CODE (optimize this, keep same interface):
#   def boxcar_matched_filter(profile: np.ndarray, width: int) -> np.ndarray:
#       n = len(profile)
#       width = min(width, n)
#       result = np.zeros(n)
#       for i in range(n):
#           total = 0.0
#           for j in range(width):
#               idx = (i + j) % n  # CIRCULAR wrap-around
#               total += profile[idx]
#           result[i] = total
#       result /= np.sqrt(width)  # Normalize by sqrt(width)
#       return result
#   
#   OPTIMIZATION APPROACH (O(n) prefix sum for circular convolution):
#   1. Create extended array: extended = np.concatenate([profile, profile[:width-1]])
#   2. Compute prefix sum: prefix = np.zeros(len(extended)+1); prefix[1:] = np.cumsum(extended)
#   3. Compute result: result[i] = (prefix[i+width] - prefix[i]) / sqrt(width) for i in range(n)
#   This handles wrap-around correctly by duplicating the first (width-1) elements.
# END-TASK-DEFINITION


def fold_timeseries(data, period_samples, n_bins=None):
    """Fold time series at specified period (supporting function)."""
    if n_bins is None:
        n_bins = period_samples
    n_samples = len(data)
    n_periods = n_samples // period_samples
    if n_periods < 2:
        return np.zeros(n_bins)
    truncated = data[: n_periods * period_samples]
    reshaped = truncated.reshape(n_periods, period_samples)
    folded = np.sum(reshaped, axis=0)
    folded /= np.sqrt(n_periods)
    return folded


def compute_profile_snr(profile):
    """Compute S/N of a folded profile (supporting function)."""
    if len(profile) < 8:
        return 0.0
    sorted_profile = np.sort(profile)
    n_baseline = max(len(profile) // 4, 2)
    baseline = sorted_profile[:n_baseline]
    mean_baseline = np.mean(baseline)
    std_baseline = np.std(baseline)
    if std_baseline < 1e-10:
        return 0.0
    return (np.max(profile) - mean_baseline) / std_baseline


# EVOLVE-BLOCK-START
def boxcar_matched_filter(profile: np.ndarray, width: int) -> np.ndarray:
    """
    Apply boxcar matched filter to a folded pulse profile.

    Performs circular convolution with a boxcar kernel of the given width.
    Used in FFA pulsar searches to maximize S/N for box-shaped pulses.

    Args:
        profile: 1D array of folded pulse profile values
        width: Width of boxcar kernel in bins (1 to len(profile)//2)

    Returns:
        Filtered profile with same length as input, normalized by sqrt(width)

    Performance: Current O(n * width), target O(n)
    """
{hint_section}
    n = len(profile)
    width = min(width, n)

    # Baseline implementation: direct convolution with wrap-around
    # This is O(n * width) - can be improved to O(n)
    result = np.zeros(n)
    for i in range(n):
        total = 0.0
        for j in range(width):
            idx = (i + j) % n
            total += profile[idx]
        result[i] = total

    # Normalize for unit response to unit impulse
    result /= np.sqrt(width)

    return result
# EVOLVE-BLOCK-END


def main():
    """Test the boxcar filter on synthetic data."""
    np.random.seed(42)
    
    # Generate test profile with synthetic pulse
    profile = np.random.randn(128)
    profile[60:68] += 5.0  # Add pulse
    
    print("Testing boxcar_matched_filter...")
    print(f"Profile shape: {{profile.shape}}")
    
    for width in [1, 4, 8, 16]:
        filtered = boxcar_matched_filter(profile, width)
        snr = compute_profile_snr(filtered)
        print(f"  Width {{width:2d}}: S/N = {{snr:.1f}}")
    
    print("\\nFilter test complete.")


if __name__ == "__main__":
    main()
'''


async def run_ffa_demo(enable_hints: bool = False):
    """Run the FFA evolution demo."""
    logger.info("=" * 60)
    logger.info("  FFA Boxcar Filter Evolution Demo")
    logger.info("  Using AlphaEvolve LLM-Based Optimization")
    logger.info("=" * 60)
    
    if enable_hints:
        logger.info("📝 Hints ENABLED (advanced optimization guidance)")
    else:
        logger.info("📝 Hints DISABLED (pure LLM discovery)")
    
    # 1. Initialize Client
    client = ReferenceClient()
    logger.info("\nInitializing LSP Client...")
    
    client.lsp_server_path = 'lsp_server.server'
    
    if await client.initialize():
        logger.info("✅ Client initialized successfully.")
    else:
        logger.error("❌ Failed to initialize client.")
        return

    # 2. Create the FFA document
    document_uri = "file:///ffa_boxcar_filter.py"
    document_content = get_ffa_document_content(enable_hints)
    
    # 3. Open the document
    logger.info(f"\nOpening document: {document_uri}")
    await client._send_notification('textDocument/didOpen', {
        'textDocument': {
            'uri': document_uri,
            'languageId': 'python',
            'version': 1,
            'text': document_content
        }
    })
    
    # 4. Request Code Actions for the EVOLVE-BLOCK (lines ~65-90)
    logger.info("Scanning for EVOLVE-BLOCK...")
    actions = await client.request_code_actions(document_uri, 65, 95)
    
    if not actions:
        logger.warning("No code actions found. Check EVOLVE-BLOCK markers.")
        client.shutdown()
        return

    logger.info(f"✅ Found {len(actions)} code actions:")
    evolve_action = None
    for action in actions:
        title = action.get('title', 'Unknown')
        logger.info(f"  - {title}")
        if "Evolve" in title:
            evolve_action = action

    if not evolve_action:
        logger.warning("No evolution action found.")
        client.shutdown()
        return

    # 5. Execute "Start Evolution"
    command = evolve_action['command']
    cmd_name = command['command']
    cmd_args = command['arguments']
    
    logger.info(f"\n🚀 Starting evolution: {cmd_name}")
    logger.info("   Target function: boxcar_matched_filter")
    logger.info("   Objective: O(n) complexity while maintaining accuracy")
    
    response = await client.execute_command(cmd_name, cmd_args)
    logger.info(f"Server response: {json.dumps(response, indent=2)}")
    
    if response.get('status') != 'started':
        logger.error(f"Failed to start evolution: {response}")
        client.shutdown()
        return
    
    evolution_id = response['evolution_id']
    block_id = cmd_args[1]
    
    # 6. Monitor Progress
    logger.info("\n⏳ Evolution in progress...")
    logger.info("   (This may take several minutes depending on LLM response time)")
    
    max_wait = 900  # 15 minutes
    poll_interval = 2.0
    elapsed = 0
    last_message = ""
    
    while elapsed < max_wait:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        
        try:
            progress = await client.execute_command(
                "alphaevolve.showProgress", 
                [document_uri, block_id]
            )
        except Exception as e:
            logger.warning(f"Progress check failed: {e}")
            continue
        
        status = progress.get('status', 'unknown')
        current_progress = progress.get('current_progress', {})
        message = current_progress.get('message', '')
        
        # Only log if message changed
        if message and message != last_message:
            logger.info(f"  [{elapsed:.0f}s] {message}")
            last_message = message
        
        if status == 'complete':
            logger.info("\n✅ Evolution completed!")
            break
        elif status in ['failed', 'stopped']:
            logger.warning(f"Evolution ended with status: {status}")
            break
    
    if elapsed >= max_wait:
        logger.warning("Evolution timed out after 15 minutes.")
    
    # 7. Display Results
    final_progress = await client.execute_command(
        "alphaevolve.showProgress", 
        [document_uri, block_id]
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("  EVOLUTION RESULTS")
    logger.info("=" * 60)
    
    results = final_progress.get('results', {})
    
    if results.get('best_code'):
        logger.info(f"Status: {final_progress.get('status')}")
        logger.info(f"Best Score: {results.get('best_score', 'N/A')}")
        logger.info(f"Generations: {results.get('generations', 'N/A')}")
        logger.info(f"Mode: {results.get('mode', 'real')}")
        
        logger.info("\n🧬 Evolved Code:")
        print("-" * 60)
        print(results.get('best_code'))
        print("-" * 60)
        
        # Show improvement metrics if available
        if 'speedup' in results:
            logger.info(f"\n📈 Performance Improvement:")
            logger.info(f"   Speedup: {results.get('speedup', 0):.1f}x")
            logger.info(f"   SNR Error: {results.get('snr_error', 0):.4f}")
    else:
        logger.info(f"Status: {final_progress.get('status')}")
        logger.info("No evolved code produced.")
        if 'error' in results:
            logger.error(f"Error: {results.get('error')}")

    # 8. Shutdown
    logger.info("\nShutting down client...")
    client.shutdown()
    
    logger.info("\n" + "=" * 60)
    logger.info("  FFA Evolution Demo Complete")
    logger.info("=" * 60)


def main():
    """Main entry point with argument parsing and API key check."""
    parser = argparse.ArgumentParser(
        description="FFA Boxcar Filter Evolution Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_ffa_evolution.py
    python demo_ffa_evolution.py --enable-hints
    
    OPENAI_API_KEY=sk-xxx python demo_ffa_evolution.py
"""
    )
    parser.add_argument(
        "--enable-hints",
        action="store_true",
        help="Enable optimization hints in the EVOLVE-BLOCK docstring"
    )
    
    args = parser.parse_args()
    
    # Check for API keys
    api_key = check_api_keys()
    if api_key is None:
        print(API_KEY_ERROR_MESSAGE, file=sys.stderr)
        sys.exit(1)
    
    logger.info(f"Using API key: {api_key}")
    
    # Run the demo
    asyncio.run(run_ffa_demo(enable_hints=args.enable_hints))


if __name__ == "__main__":
    main()
