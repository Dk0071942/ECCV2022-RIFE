"""
GPU Memory Monitoring and Management Utilities for RIFE

This module provides comprehensive GPU memory monitoring, automatic cleanup,
and fallback mechanisms to prevent out-of-memory errors during interpolation.

Features:
- Real-time GPU memory usage tracking
- Automatic memory cleanup between operations
- Memory pressure detection and warnings
- Automatic fallback to hierarchical mode on high memory usage
- Memory usage history and analytics
"""

import torch
import gc
import time
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    used_mb: float
    total_mb: float
    free_mb: float
    reserved_mb: float
    utilization_percent: float


class GPUMemoryMonitor:
    """
    Comprehensive GPU memory monitoring and management system.
    
    Provides real-time memory tracking, automatic cleanup, and intelligent
    fallback recommendations to prevent OOM errors.
    """
    
    def __init__(self, enable_logging: bool = True, log_file: Optional[Path] = None):
        """
        Initialize the GPU memory monitor.
        
        Args:
            enable_logging: Enable memory usage logging
            log_file: Path to log file (optional)
        """
        self.enable_logging = enable_logging
        self.log_file = log_file
        self.memory_history: List[MemorySnapshot] = []
        self.cleanup_count = 0
        
        # Memory thresholds (in percentage)
        self.warning_threshold = 75.0  # Warn when >75% used
        self.critical_threshold = 85.0  # Critical when >85% used
        self.emergency_threshold = 95.0  # Emergency when >95% used
        
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory information.
        
        Returns:
            Dictionary with memory statistics in MB and percentages
        """
        if not torch.cuda.is_available():
            return {
                "used_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0,
                "reserved_mb": 0.0, "utilization_percent": 0.0,
                "device_available": False
            }
        
        # Get memory statistics
        used_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        
        # Convert to MB
        used_mb = used_bytes / (1024 ** 2)
        reserved_mb = reserved_bytes / (1024 ** 2)
        total_mb = total_bytes / (1024 ** 2)
        free_mb = total_mb - used_mb
        
        utilization_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
        
        return {
            "used_mb": round(used_mb, 2),
            "total_mb": round(total_mb, 2),
            "free_mb": round(free_mb, 2),
            "reserved_mb": round(reserved_mb, 2),
            "utilization_percent": round(utilization_percent, 2),
            "device_available": True
        }
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        info = self.get_memory_info()
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            used_mb=info["used_mb"],
            total_mb=info["total_mb"],
            free_mb=info["free_mb"],
            reserved_mb=info["reserved_mb"],
            utilization_percent=info["utilization_percent"]
        )
        
        if self.enable_logging:
            self.memory_history.append(snapshot)
            # Keep only last 100 snapshots
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-100:]
                
        return snapshot
    
    def cleanup_gpu_memory(self, aggressive: bool = False) -> Dict[str, float]:
        """
        Perform GPU memory cleanup.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
            
        Returns:
            Memory info before and after cleanup
        """
        before = self.get_memory_info()
        
        # Standard cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if aggressive:
            # Aggressive cleanup - multiple rounds
            for _ in range(3):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.1)  # Brief pause between cleanup rounds
        
        after = self.get_memory_info()
        self.cleanup_count += 1
        
        freed_mb = before["used_mb"] - after["used_mb"]
        
        if self.enable_logging and freed_mb > 0:
            print(f"🧹 GPU Memory cleaned: {freed_mb:.1f}MB freed "
                  f"({before['used_mb']:.1f}MB → {after['used_mb']:.1f}MB)")
        
        return {
            "before": before,
            "after": after,
            "freed_mb": round(freed_mb, 2)
        }
    
    def check_memory_pressure(self) -> Tuple[str, str]:
        """
        Check current memory pressure level.
        
        Returns:
            Tuple of (pressure_level, message)
            Levels: "normal", "warning", "critical", "emergency"
        """
        info = self.get_memory_info()
        
        if not info["device_available"]:
            return "unknown", "GPU not available for memory monitoring"
        
        utilization = info["utilization_percent"]
        
        if utilization >= self.emergency_threshold:
            return "emergency", f"⚠️ EMERGENCY: GPU memory at {utilization:.1f}% - Immediate action required!"
        elif utilization >= self.critical_threshold:
            return "critical", f"🚨 CRITICAL: GPU memory at {utilization:.1f}% - Consider hierarchical mode"
        elif utilization >= self.warning_threshold:
            return "warning", f"⚠️ WARNING: GPU memory at {utilization:.1f}% - Monitor usage closely"
        else:
            return "normal", f"✅ NORMAL: GPU memory at {utilization:.1f}% - Operating normally"
    
    def recommend_hierarchical_mode(self, exp_value: int, frame_size_mb: float = 3.0) -> Tuple[bool, str]:
        """
        Recommend whether to use hierarchical mode based on memory pressure.
        
        Args:
            exp_value: Standard interpolation exponent
            frame_size_mb: Estimated size per frame in MB
            
        Returns:
            Tuple of (should_use_hierarchical, reason)
        """
        info = self.get_memory_info()
        
        if not info["device_available"]:
            return True, "GPU not available - hierarchical mode safer"
        
        # Estimate memory needed for standard mode
        estimated_frames = (2 ** exp_value) - 1
        estimated_memory_mb = estimated_frames * frame_size_mb
        
        available_mb = info["free_mb"]
        utilization = info["utilization_percent"]
        
        # Check if estimated memory exceeds available
        if estimated_memory_mb > available_mb * 0.8:  # Leave 20% buffer
            return True, f"Estimated {estimated_memory_mb:.1f}MB needed, only {available_mb:.1f}MB available"
        
        # Check current memory pressure
        if utilization > self.critical_threshold:
            return True, f"High memory pressure ({utilization:.1f}%) - hierarchical recommended"
        
        # Check if exp_value is high
        if exp_value >= 4:
            return True, f"High interpolation factor (exp={exp_value}) - hierarchical recommended"
        
        return False, f"Standard mode should work (estimated {estimated_memory_mb:.1f}MB, {available_mb:.1f}MB available)"
    
    def get_memory_report(self) -> str:
        """Generate a comprehensive memory usage report."""
        current = self.get_memory_info()
        pressure_level, pressure_msg = self.check_memory_pressure()
        
        report = [
            "=== GPU Memory Report ===",
            f"Device Available: {current['device_available']}",
            f"Total Memory: {current['total_mb']:.1f}MB",
            f"Used Memory: {current['used_mb']:.1f}MB ({current['utilization_percent']:.1f}%)",
            f"Free Memory: {current['free_mb']:.1f}MB",
            f"Reserved Memory: {current['reserved_mb']:.1f}MB",
            f"Memory Pressure: {pressure_level.upper()}",
            f"Status: {pressure_msg}",
            f"Cleanup Operations: {self.cleanup_count}",
        ]
        
        if len(self.memory_history) > 1:
            peak_usage = max(snapshot.utilization_percent for snapshot in self.memory_history)
            avg_usage = sum(snapshot.utilization_percent for snapshot in self.memory_history) / len(self.memory_history)
            report.extend([
                f"Peak Usage: {peak_usage:.1f}%",
                f"Average Usage: {avg_usage:.1f}%",
                f"History Length: {len(self.memory_history)} snapshots"
            ])
        
        return "\n".join(report)
    
    def save_memory_log(self, filepath: Optional[Path] = None) -> bool:
        """
        Save memory usage history to a JSON file.
        
        Args:
            filepath: Path to save file (uses self.log_file if not provided)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.memory_history:
            return False
        
        save_path = filepath or self.log_file
        if not save_path:
            return False
        
        try:
            data = {
                "snapshots": [
                    {
                        "timestamp": snap.timestamp,
                        "used_mb": snap.used_mb,
                        "total_mb": snap.total_mb,
                        "free_mb": snap.free_mb,
                        "reserved_mb": snap.reserved_mb,
                        "utilization_percent": snap.utilization_percent
                    }
                    for snap in self.memory_history
                ],
                "cleanup_count": self.cleanup_count,
                "thresholds": {
                    "warning": self.warning_threshold,
                    "critical": self.critical_threshold,
                    "emergency": self.emergency_threshold
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            if self.enable_logging:
                print(f"Failed to save memory log: {e}")
            return False


# Global memory monitor instance
_global_monitor: Optional[GPUMemoryMonitor] = None


def get_global_monitor() -> GPUMemoryMonitor:
    """Get or create the global memory monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = GPUMemoryMonitor()
    return _global_monitor


def monitor_memory_usage(operation_name: str = "operation"):
    """
    Decorator to monitor memory usage around an operation.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            
            # Take snapshot before operation
            before = monitor.take_snapshot()
            
            try:
                result = func(*args, **kwargs)
                
                # Take snapshot after operation
                after = monitor.take_snapshot()
                
                memory_delta = after.used_mb - before.used_mb
                if monitor.enable_logging:
                    print(f"📊 {operation_name} memory usage: {memory_delta:+.1f}MB "
                          f"({before.used_mb:.1f}MB → {after.used_mb:.1f}MB)")
                
                return result
                
            except Exception as e:
                # Take snapshot on error for debugging
                error_snapshot = monitor.take_snapshot()
                if monitor.enable_logging:
                    print(f"❌ {operation_name} failed with memory at {error_snapshot.utilization_percent:.1f}%")
                raise e
        
        return wrapper
    return decorator


def cleanup_on_low_memory(threshold: float = 85.0):
    """
    Decorator to automatically cleanup memory if usage exceeds threshold.
    
    Args:
        threshold: Memory utilization threshold for cleanup (percentage)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            
            # Check memory before operation
            info = monitor.get_memory_info()
            if info["utilization_percent"] > threshold:
                monitor.cleanup_gpu_memory(aggressive=True)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator