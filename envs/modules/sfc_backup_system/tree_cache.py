# sfc_backup_system/tree_cache.py
import logging

logger = logging.getLogger(__name__)


class TreeCache:
    """
    Simple cache for tree path costs with invalidation by version.
    Stores keys as (src, dst, version).
    """

    def __init__(self, max_items: int = 1000):
        self.cache = {}
        self.max_items = max_items
        self.version = 0

    def invalidate(self):
        self.cache.clear()
        self.version += 1
        logger.debug(f"TreeCache invalidated, version={self.version}")

    def get(self, src, dst):
        return self.cache.get((src, dst, self.version))

    def set(self, src, dst, value):
        key = (src, dst, self.version)
        self.cache[key] = value
        if len(self.cache) > self.max_items:
            # purge approx half oldest keys
            keys = list(self.cache.keys())[: self.max_items // 2]
            for k in keys:
                del self.cache[k]
