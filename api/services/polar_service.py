from pymilvus import Collection, connections
from typing import List, Dict, Any
from config.settings import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, TOP_K, OBJECT_DATABASE
import polars as pl
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
    
    def search_object(self, objects: List[str]):
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

        result = self.database.filter(combined_condition)

        if result.is_empty():
            return [] 

        return result[["filepath", "frame_id", "video_id"]]
