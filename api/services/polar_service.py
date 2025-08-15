import asyncio
from typing import List
import polars as pl
from config.settings import OBJECT_DATABASE, TOP_K
import re

class PolarService:
    def __init__(self):
        print("Init Polar Service...")
        self.database = self._load_database()

    def _load_database(self):
        try:
            return pl.read_parquet(OBJECT_DATABASE)
        except Exception as e:
            # raise ConnectionError(f"Failed to load database: {str(e)}")
            return None
        
    def get_object_by_frame(self, video_id: str, frame_id: str) -> dict:
        """
        Get all object information for a specific video_id and frame_id,
        excluding the 'filepath' column.

        Returns:
            dict with column:value pairs or None if not found
        """
        try:
            df = pl.read_parquet(OBJECT_DATABASE)
            result = (
                df.filter(
                    (pl.col("video_id") == video_id) &
                    (pl.col("frame_id") == frame_id)
                )
                .select(pl.all().exclude("filepath"))
            )

            if result.is_empty():
                return None

            return result.to_dicts()[0]  # Return first matching row as dict
        except Exception as e:
            raise RuntimeError(f"Error getting object data: {str(e)}")

    async def search_object(self, objects: List[str]):
        query_str = objects[0]
        object_filters = query_str.strip().split(" ")

        conditions = []
        for expr in object_filters:
            match = re.match(r"(\w+)([<>=]+)(\d+)", expr)
            if match:
                obj, op, value = match.groups()
                col = pl.col(obj)
                value = int(value)

                if op == "=":
                    conditions.append(col == value)
                elif op == ">":
                    conditions.append(col > value)
                elif op == ">=":
                    conditions.append(col >= value)
                elif op == "<":
                    conditions.append(col < value)
                elif op == "<=":
                    conditions.append(col <= value)
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            else:
                raise ValueError(f"Invalid object filter format: {expr}")

        if not conditions:
            raise ValueError("No valid object filters provided.")

        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition &= cond

        def blocking_filter():
            # lazy read & filter với giới hạn
            lazy_df = pl.scan_parquet(OBJECT_DATABASE)
            filtered = lazy_df.filter(combined_condition).select(["filepath", "frame_id", "video_id"])
            return filtered.collect()

        result = await asyncio.to_thread(blocking_filter)

        if result.is_empty():
            return []

        return result
