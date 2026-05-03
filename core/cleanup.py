import os
import time
import logging

logger = logging.getLogger("core.cleanup")

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def cleanup_old_files(directory_name, max_age_seconds=600):
    """
    Deletes files in a directory older than max_age_seconds.
    Supports absolute and relative paths.
    """
    directory = directory_name if os.path.isabs(directory_name) else os.path.join(_BASE_DIR, directory_name)
    directory = os.path.abspath(directory)
    
    if not os.path.exists(directory):
        logger.warning(f"Cleanup: Directory {directory} does not exist.")
        return
    
    now = time.time()
    count = 0
    try:
        for f in os.listdir(directory):
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath):
                if os.stat(filepath).st_mtime < now - max_age_seconds:
                    try:
                        os.remove(filepath)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {filepath}: {e}")
        if count > 0:
            logger.info(f"Cleaned up {count} old files in {directory}")
    except Exception as e:
        logger.error(f"Cleanup error in {directory}: {e}")
