"""
Resource cleanup utilities for optstop package.

This module provides robust cleanup mechanisms for temporary directories,
worker processes, and system resources to prevent resource leaks and
ensure clean termination of parallel processing tasks.
"""

import os
import shutil
import atexit
import signal
import logging
import tempfile
import threading
from typing import List, Optional, Set
from contextlib import contextmanager
import time


class ResourceCleanupManager:
    """
    Centralized cleanup manager for tracking and cleaning up temporary resources.

    Provides robust cleanup with verification, logging, and signal handling
    to ensure resources are properly released even under abnormal termination.
    """

    def __init__(self, logger_name: str = 'optstop.cleanup'):
        self._temp_dirs: Set[str] = set()
        self._cleanup_registered: bool = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger(logger_name)

    def register_temp_dir(self, path: str) -> None:
        """
        Register a temporary directory for cleanup.

        Args:
            path: Path to temporary directory to track for cleanup
        """
        with self._lock:
            if os.path.exists(path):
                self._temp_dirs.add(path)
                self.logger.debug(f"Registered temp directory: {path}")

                # Register cleanup handlers on first use
                if not self._cleanup_registered:
                    self._register_cleanup_handlers()
                    self._cleanup_registered = True

    def unregister_temp_dir(self, path: str) -> None:
        """
        Unregister a temporary directory (already cleaned up).

        Args:
            path: Path to remove from cleanup tracking
        """
        with self._lock:
            self._temp_dirs.discard(path)
            self.logger.debug(f"Unregistered temp directory: {path}")

    def cleanup_temp_dir(self, path: str) -> bool:
        """
        Clean up a specific temporary directory with verification.

        Args:
            path: Path to directory to clean up

        Returns:
            bool: True if cleanup successful, False otherwise
        """
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                # Verify cleanup success
                if not os.path.exists(path):
                    self.logger.info(f"Successfully cleaned up: {path}")
                    with self._lock:
                        self._temp_dirs.discard(path)
                    return True
                else:
                    self.logger.error(f"Cleanup verification failed for: {path}")
                    return False
            else:
                # Already cleaned up
                with self._lock:
                    self._temp_dirs.discard(path)
                return True
        except Exception as e:
            self.logger.error(f"Cleanup failed for {path}: {e}")
            return False

    def cleanup_all(self) -> int:
        """
        Clean up all registered temporary directories.

        Returns:
            int: Number of directories successfully cleaned up
        """
        cleaned_count = 0
        with self._lock:
            temp_dirs_copy = self._temp_dirs.copy()

        for temp_dir in temp_dirs_copy:
            if self.cleanup_temp_dir(temp_dir):
                cleaned_count += 1

        remaining = len(self._temp_dirs)
        if remaining > 0:
            self.logger.warning(f"Failed to clean up {remaining} directories")
        else:
            self.logger.info(f"Successfully cleaned up all {cleaned_count} directories")

        return cleaned_count

    def _register_cleanup_handlers(self) -> None:
        """Register atexit and signal handlers for cleanup."""
        atexit.register(self._emergency_cleanup)

        # Register signal handlers for graceful cleanup
        try:
            signal.signal(signal.SIGTERM, self._signal_cleanup)
            signal.signal(signal.SIGINT, self._signal_cleanup)
            self.logger.debug("Registered signal handlers for cleanup")
        except (AttributeError, ValueError):
            # Some signals may not be available on all platforms
            self.logger.debug("Some signal handlers not available on this platform")

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup called via atexit."""
        if self._temp_dirs:
            self.logger.info(f"Emergency cleanup of {len(self._temp_dirs)} directories")
            self.cleanup_all()

    def _signal_cleanup(self, signum, frame) -> None:
        """Signal handler for cleanup on termination signals."""
        self.logger.info(f"Received signal {signum}, performing cleanup")
        self.cleanup_all()
        # Re-raise the signal for proper termination
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# Global cleanup manager instance
_global_cleanup_manager: Optional[ResourceCleanupManager] = None


def get_cleanup_manager() -> ResourceCleanupManager:
    """
    Get the global cleanup manager instance.

    Returns:
        ResourceCleanupManager: Global cleanup manager
    """
    global _global_cleanup_manager
    if _global_cleanup_manager is None:
        _global_cleanup_manager = ResourceCleanupManager()
    return _global_cleanup_manager


@contextmanager
def managed_temp_dir(prefix: str = 'optstop_', suffix: str = ''):
    """
    Context manager for creating and cleaning up temporary directories.

    Args:
        prefix: Prefix for temporary directory name
        suffix: Suffix for temporary directory name

    Yields:
        str: Path to temporary directory

    Example:
        with managed_temp_dir('optstop_workers_') as temp_dir:
            # Use temp_dir
            pass
        # temp_dir is automatically cleaned up
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
    cleanup_manager = get_cleanup_manager()
    cleanup_manager.register_temp_dir(temp_dir)

    try:
        yield temp_dir
    finally:
        # Explicit cleanup with verification
        success = cleanup_manager.cleanup_temp_dir(temp_dir)
        if not success:
            # Log but don't raise - emergency cleanup will handle it
            logging.getLogger('optstop.cleanup').warning(
                f"Context manager cleanup failed for {temp_dir}, "
                "registered for emergency cleanup"
            )


def create_worker_temp_dir(prefix: str, process_id: Optional[int] = None) -> str:
    """
    Create a temporary directory for worker processes with robust cleanup.

    Args:
        prefix: Prefix for directory name
        process_id: Process ID to include in directory name (uses current PID if None)

    Returns:
        str: Path to created temporary directory
    """
    if process_id is None:
        process_id = os.getpid()

    # Create highly unique directory name
    timestamp = int(time.time() * 1000000)  # microsecond precision
    import random
    random_id = random.randint(10000, 99999)
    unique_suffix = f'{process_id}_{timestamp}_{random_id}'

    temp_dir = tempfile.mkdtemp(prefix=f'{prefix}_{unique_suffix}_')

    # Register for cleanup
    cleanup_manager = get_cleanup_manager()
    cleanup_manager.register_temp_dir(temp_dir)

    return temp_dir


def register_worker_cleanup(temp_dir: str, logger_name: str = 'optstop.worker_cleanup') -> None:
    """
    Register cleanup for worker processes with enhanced verification.

    Args:
        temp_dir: Temporary directory to clean up on process exit
        logger_name: Logger name for cleanup messages
    """
    logger = logging.getLogger(logger_name)

    def verified_cleanup():
        """Cleanup function with verification and logging."""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                # Verify cleanup success
                if not os.path.exists(temp_dir):
                    logger.info(f"Worker {os.getpid()} cleanup verified: {temp_dir}")
                else:
                    logger.error(f"Worker {os.getpid()} cleanup verification failed: {temp_dir}")
            else:
                logger.debug(f"Worker {os.getpid()} temp dir already cleaned: {temp_dir}")
        except Exception as e:
            logger.error(f"Worker {os.getpid()} cleanup failed for {temp_dir}: {e}")

    # Register cleanup with atexit
    atexit.register(verified_cleanup)

    # Also register signal handlers for more robust cleanup
    try:
        def signal_cleanup(signum, frame):
            logger.info(f"Worker {os.getpid()} received signal {signum}, cleaning up {temp_dir}")
            verified_cleanup()
            # Re-raise signal for proper termination
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, signal_cleanup)
        signal.signal(signal.SIGINT, signal_cleanup)
    except (AttributeError, ValueError):
        # Signal handling not available on all platforms
        pass


def cleanup_orphaned_temp_dirs(max_age_hours: float = 24) -> int:
    """
    Clean up orphaned optstop temporary directories older than specified age.

    Args:
        max_age_hours: Maximum age in hours for temp directories before cleanup

    Returns:
        int: Number of directories cleaned up
    """
    logger = logging.getLogger('optstop.orphan_cleanup')
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        import tempfile
        temp_root = tempfile.gettempdir()

        for item in os.listdir(temp_root):
            if item.startswith('optstop_'):
                item_path = os.path.join(temp_root, item)
                try:
                    if os.path.isdir(item_path):
                        # Check age of directory
                        dir_mtime = os.path.getmtime(item_path)
                        age_seconds = current_time - dir_mtime

                        if age_seconds > max_age_seconds:
                            logger.info(f"Cleaning up orphaned directory (age: {age_seconds/3600:.1f}h): {item_path}")
                            shutil.rmtree(item_path, ignore_errors=True)

                            # Verify cleanup
                            if not os.path.exists(item_path):
                                cleaned_count += 1
                                logger.info(f"Successfully cleaned up orphaned directory: {item_path}")
                            else:
                                logger.warning(f"Failed to clean up orphaned directory: {item_path}")
                        else:
                            logger.debug(f"Keeping recent directory (age: {age_seconds/3600:.1f}h): {item_path}")

                except (OSError, IOError) as e:
                    logger.warning(f"Error processing {item_path}: {e}")

    except (OSError, IOError) as e:
        logger.error(f"Error accessing temp directory: {e}")

    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} orphaned optstop directories")

    return cleaned_count