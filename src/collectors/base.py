from abc import ABC, abstractmethod
import aiosqlite


class BaseCollector(ABC):
    """
    Abstract base class for all platform collectors.

    Each concrete subclass must:
    - Set `platform` class attribute matching sources.platform value (e.g., "telegram")
    - Implement `collect(db, source)` to fetch new content and write to raw_posts
    - Call update_source_last_cursor() after successful fetch
    """
    platform: str  # class-level attribute — matches sources.platform value

    @abstractmethod
    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        """
        Fetch new content for one source row and write to raw_posts.
        Updates last_cursor on the source after successful fetch.
        Raises on unrecoverable error; collect_all() catches per-source.

        Args:
            db: aiosqlite connection (already open)
            source: dict row from sources table — keys include id, target, last_cursor
        """
        ...
