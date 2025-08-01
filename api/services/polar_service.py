import asyncio
from typing import List
import polars as pl
from config.settings import OBJECT_DATABASE
import re

class PolarService:
    def __init__(self):
        print("Init Polar Service...")
        self.database = self._load_database()

    def _load_database(self):
        try:
            return pl.read_parquet(OBJECT_DATABASE)
        except Exception as e:
            raise ConnectionError(f"Failed to load database: {str(e)}")

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
            filtered = lazy_df.filter(combined_condition).select(["filepath", "frame_id", "video_id"]).limit(1000)
            return filtered.collect()

        result = await asyncio.to_thread(blocking_filter)

        if result.is_empty():
            return []

        return result
