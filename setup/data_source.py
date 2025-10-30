"""
Supabase data source for video generation data.
Fetches generation data (prompts, images, categories) from Supabase.
"""

import os
import threading
import logging
from typing import List, Dict, Optional
from pathlib import Path

from supabase import create_client, Client

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global Supabase client
supabase_client = None
supabase_client_lock = threading.Lock()


def get_supabase_client() -> Client:
    """Get or create a Supabase client (thread-safe singleton)."""
    global supabase_client
    if supabase_client is None:
        with supabase_client_lock:
            if supabase_client is None:
                if not config.SUPABASE_URL or not config.SUPABASE_KEY:
                    raise ValueError(
                        "Supabase URL or KEY not configured. "
                        "Set SUPABASE_URL and SUPABASE_KEY in .env file."
                    )
                supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                logger.info("Supabase client initialized")
    return supabase_client


def supabase_exec(func, retries: int = config.MAX_RETRIES):
    """
    Execute a Supabase query function with retries.

    Args:
        func: Function that takes a Supabase client and returns a query
        retries: Number of retry attempts

    Returns:
        Query results
    """
    client = get_supabase_client()
    last_exception = None

    for i in range(retries):
        try:
            return func(client).execute()
        except Exception as e:
            last_exception = e
            if i == retries - 1:
                raise
            # Simple backoff: wait i+1 seconds between retries
            import time
            logger.warning(f"Supabase query failed (attempt {i+1}/{retries}): {e}")
            time.sleep(i + 1)

    if last_exception:
        raise last_exception


class SupabaseGenerationSource:
    """
    Data source for video generations from Supabase.

    Fetches generation data including:
    - Generation ID
    - Prompt text
    - Image URL/path
    - Category
    - User ID (optional)
    - Timestamp (optional)
    """

    def __init__(
        self,
        table_name: str = config.SUPABASE_TABLE_NAME,
        image_column: str = config.SUPABASE_IMAGE_COLUMN,
        prompt_column: str = config.SUPABASE_PROMPT_COLUMN,
        category_column: str = config.SUPABASE_CATEGORY_COLUMN,
        id_column: str = config.SUPABASE_ID_COLUMN,
        user_column: str = config.SUPABASE_USER_COLUMN,
        timestamp_column: str = config.SUPABASE_TIMESTAMP_COLUMN
    ):
        """
        Initialize Supabase data source.

        Args:
            table_name: Name of the generations table
            image_column: Column name for image URL/path
            prompt_column: Column name for prompt text
            category_column: Column name for category
            id_column: Column name for generation ID
            user_column: Column name for user ID
            timestamp_column: Column name for timestamp
        """
        self.table_name = table_name
        self.image_column = image_column
        self.prompt_column = prompt_column
        self.category_column = category_column
        self.id_column = id_column
        self.user_column = user_column
        self.timestamp_column = timestamp_column

        self.client = get_supabase_client()
        logger.info(f"SupabaseGenerationSource initialized for table '{table_name}'")

    def get_generations(
        self,
        limit: int = 1000,
        offset: int = 0,
        category: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch generations from Supabase.

        Args:
            limit: Maximum number of generations to fetch
            offset: Offset for pagination
            category: Filter by category (optional)
            user_id: Filter by user ID (optional)

        Returns:
            List of generation dictionaries
        """
        try:
            # Build query
            query = self.client.table(self.table_name).select("*")

            # Apply filters
            if category:
                query = query.eq(self.category_column, category)
            if user_id:
                query = query.eq(self.user_column, user_id)

            # Apply pagination
            query = query.range(offset, offset + limit - 1)

            # Execute
            response = query.execute()

            # Transform to standard format
            generations = []
            for row in response.data:
                gen = {
                    'generation_id': row.get(self.id_column),
                    'prompt': row.get(self.prompt_column),
                    'image_url': row.get(self.image_column),
                    'category': row.get(self.category_column, 'unknown'),
                    'user_id': row.get(self.user_column),
                    'timestamp': row.get(self.timestamp_column)
                }
                generations.append(gen)

            logger.info(f"Fetched {len(generations)} generations (offset={offset}, limit={limit})")
            return generations

        except Exception as e:
            logger.error(f"Failed to fetch generations: {e}")
            raise

    def get_generations_by_category(
        self,
        category: str,
        limit: int = 10000
    ) -> List[Dict]:
        """
        Fetch generations for a specific category.

        Args:
            category: Category to filter by
            limit: Maximum number to fetch

        Returns:
            List of generation dictionaries
        """
        return self.get_generations(limit=limit, category=category)

    def get_all_categories(self) -> List[str]:
        """
        Get list of all unique categories in the database.

        Returns:
            List of category names
        """
        try:
            query = self.client.table(self.table_name).select(self.category_column)
            response = query.execute()

            categories = set()
            for row in response.data:
                cat = row.get(self.category_column)
                if cat:
                    categories.add(cat)

            logger.info(f"Found {len(categories)} unique categories: {sorted(categories)}")
            return sorted(categories)

        except Exception as e:
            logger.error(f"Failed to fetch categories: {e}")
            return config.CONTENT_CATEGORIES  # Fallback to config

    def get_generation_count(self, category: Optional[str] = None) -> int:
        """
        Get total count of generations.

        Args:
            category: Filter by category (optional)

        Returns:
            Count of generations
        """
        try:
            query = self.client.table(self.table_name).select("*", count="exact")

            if category:
                query = query.eq(self.category_column, category)

            response = query.execute()
            count = response.count if hasattr(response, 'count') else len(response.data)

            logger.info(f"Total generations{f' in {category}' if category else ''}: {count}")
            return count

        except Exception as e:
            logger.error(f"Failed to get generation count: {e}")
            return 0

    def get_user_generations(
        self,
        user_id: str,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch generations for a specific user.

        Args:
            user_id: User ID to filter by
            limit: Maximum number to fetch
            start_date: Filter by start date (optional, ISO format)
            end_date: Filter by end date (optional, ISO format)

        Returns:
            List of generation dictionaries
        """
        try:
            query = self.client.table(self.table_name).select("*")
            query = query.eq(self.user_column, user_id)

            # Apply date filters if provided
            if start_date:
                query = query.gte(self.timestamp_column, start_date)
            if end_date:
                query = query.lte(self.timestamp_column, end_date)

            # Order by timestamp descending (most recent first)
            query = query.order(self.timestamp_column, desc=True)
            query = query.limit(limit)

            response = query.execute()

            generations = []
            for row in response.data:
                gen = {
                    'generation_id': row.get(self.id_column),
                    'prompt': row.get(self.prompt_column),
                    'image_url': row.get(self.image_column),
                    'category': row.get(self.category_column, 'unknown'),
                    'user_id': row.get(self.user_column),
                    'timestamp': row.get(self.timestamp_column)
                }
                generations.append(gen)

            logger.info(f"Fetched {len(generations)} generations for user {user_id}")
            return generations

        except Exception as e:
            logger.error(f"Failed to fetch user generations: {e}")
            raise

    def sample_diverse_generations(
        self,
        total: int = 100000,
        balance_categories: bool = True
    ) -> List[Dict]:
        """
        Sample diverse generations for building base map.

        Args:
            total: Total number of generations to sample
            balance_categories: Whether to balance across categories

        Returns:
            List of generation dictionaries
        """
        logger.info(f"Sampling {total} diverse generations...")

        if not balance_categories:
            # Simple random sampling
            return self.get_generations(limit=total)

        # Balance across categories
        categories = self.get_all_categories()
        per_category = total // len(categories)

        all_generations = []
        for category in categories:
            logger.info(f"Sampling {per_category} generations from category '{category}'")
            gens = self.get_generations_by_category(category, limit=per_category)
            all_generations.extend(gens)

        logger.info(f"Sampled {len(all_generations)} total generations")
        return all_generations


# Example usage
if __name__ == "__main__":
    try:
        source = SupabaseGenerationSource()

        # Test: Get categories
        print("\n=== Categories ===")
        categories = source.get_all_categories()
        print(f"Categories: {categories}")

        # Test: Get counts
        print("\n=== Counts ===")
        total_count = source.get_generation_count()
        print(f"Total generations: {total_count}")

        for cat in categories:
            count = source.get_generation_count(category=cat)
            print(f"{cat}: {count}")

        # Test: Fetch sample generations
        print("\n=== Sample Generations ===")
        sample = source.get_generations(limit=5)
        for i, gen in enumerate(sample, 1):
            print(f"\n{i}. ID: {gen['generation_id']}")
            print(f"   Prompt: {gen['prompt'][:50]}...")
            print(f"   Category: {gen['category']}")
            print(f"   Image: {gen['image_url']}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set SUPABASE_URL and SUPABASE_KEY in .env file")
