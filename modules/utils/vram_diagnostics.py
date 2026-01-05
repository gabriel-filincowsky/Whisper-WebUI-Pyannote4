"""
VRAM Diagnostic Utility

Provides logging functions to trace GPU memory usage at key points in the pipeline.
This diagnostic tool was used to identify and resolve memory surge issues.

To enable diagnostic logging, set the environment variable:
    export VRAM_DIAGNOSTICS_ENABLED=1

Or modify this file to set VRAM_DIAGNOSTICS_ENABLED = True
"""

import torch
import time
import os
from functools import wraps

# Enable/disable diagnostic logging globally
# Can be enabled via environment variable: VRAM_DIAGNOSTICS_ENABLED=1
VRAM_DIAGNOSTICS_ENABLED = os.environ.get("VRAM_DIAGNOSTICS_ENABLED", "0").lower() in ("1", "true", "yes")

def get_vram_stats() -> dict:
    """Get current VRAM statistics."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "max_allocated_mb": round(max_allocated, 2)
    }

def log_vram(checkpoint: str, extra_info: str = ""):
    """
    Log VRAM usage at a specific checkpoint.
    
    Parameters
    ----------
    checkpoint : str
        Name of the checkpoint (e.g., "diarization_start", "model_loaded")
    extra_info : str
        Additional context to log
    """
    if not VRAM_DIAGNOSTICS_ENABLED:
        return
    
    stats = get_vram_stats()
    timestamp = time.strftime("%H:%M:%S")
    
    msg = f"[VRAM-DIAG {timestamp}] {checkpoint}"
    if extra_info:
        msg += f" | {extra_info}"
    msg += f" | Allocated: {stats['allocated_mb']} MB | Reserved: {stats['reserved_mb']} MB | Peak: {stats['max_allocated_mb']} MB"
    
    print(msg, flush=True)

def log_vram_delta(checkpoint: str, prev_stats: dict, extra_info: str = ""):
    """
    Log VRAM usage change from a previous checkpoint.
    
    Parameters
    ----------
    checkpoint : str
        Name of the checkpoint
    prev_stats : dict
        Previous VRAM stats from get_vram_stats()
    extra_info : str
        Additional context to log
    """
    if not VRAM_DIAGNOSTICS_ENABLED:
        return
    
    current = get_vram_stats()
    delta_allocated = current['allocated_mb'] - prev_stats['allocated_mb']
    delta_reserved = current['reserved_mb'] - prev_stats['reserved_mb']
    
    timestamp = time.strftime("%H:%M:%S")
    
    sign_alloc = "+" if delta_allocated >= 0 else ""
    sign_res = "+" if delta_reserved >= 0 else ""
    
    msg = f"[VRAM-DIAG {timestamp}] {checkpoint}"
    if extra_info:
        msg += f" | {extra_info}"
    msg += f" | Allocated: {current['allocated_mb']} MB ({sign_alloc}{round(delta_allocated, 2)} MB)"
    msg += f" | Reserved: {current['reserved_mb']} MB ({sign_res}{round(delta_reserved, 2)} MB)"
    msg += f" | Peak: {current['max_allocated_mb']} MB"
    
    print(msg, flush=True)

def reset_peak_stats():
    """Reset peak memory statistics for fresh tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        log_vram("peak_stats_reset", "Peak memory tracking reset")
