import os
import json
from pathlib import Path
from typing import List, Dict, Generator, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import concurrent.futures
from dataclasses import dataclass

ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")


@dataclass
class SearchResult:
    video_id: str
    frame_id: str
    score: float
    filepath: str
    caption: str   # add this

    @classmethod
    def from_es_hit(cls, hit: Dict) -> 'SearchResult':
        source = hit['_source']
        frame_id = source['frame_id']
        video_id = source['video_id']
        caption = source['caption']
        filepath = str(ROOT / video_id / f"keyframes/keyframe_{frame_id}.webp")
        return cls(video_id=video_id, frame_id=frame_id, score=hit['_score'], filepath=filepath, caption=caption)


class ElasticsearchClient:
    """Client for indexing and searching Elasticsearch"""

    def __init__(self):
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "captions2025")
        self.es = Elasticsearch(
            hosts=[f"http://{os.getenv('ELASTICSEARCH_HOST', 'elasticsearch')}:{os.getenv('ELASTICSEARCH_PORT', '9200')}"],
            verify_certs=False,
            request_timeout=300,
            max_retries=3,
            retry_on_timeout=True,
            basic_auth=(os.getenv("ELASTICSEARCH_USER", ""), os.getenv("ELASTICSEARCH_PASS", ""))
            if os.getenv("ELASTICSEARCH_USER") else None
        )

    def check_connection(self) -> bool:
        return self.es.ping()

    def get_index_settings(self) -> Dict:
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.mapping.ignore_malformed": True,
                "analysis": {
                    "analyzer": {
                        "vietnamese": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "caption": {
                        "type": "text",
                        "analyzer": "vietnamese",
                        "fields": {
                            "raw": {"type": "keyword"},
                            "exact": {"type": "text", "analyzer": "standard"}
                        }
                    },
                    "video_id": {"type": "keyword"},
                    "frame_id": {"type": "keyword"}
                }
            }
        }

    def create_index(self, recreate: bool = False):
        if recreate and self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name, request_timeout=120, master_timeout="120s")
            print(f"🧹 Deleted existing index: {self.index_name}")
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=self.get_index_settings())
            print(f"✅ Created index: {self.index_name}")

    def _process_files(self, video_dir: Path) -> Generator[Dict, None, None]:
        caption_file = video_dir / "scene_vi_blip.json"
        if not caption_file.exists():
            return

        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)

        for frame_id, caption_text in caption_data.items():
            if not caption_text.strip():
                continue
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "caption": caption_text,
                "video_id": video_dir.name,
                "frame_id": frame_id
            }

    def index_dataset(self, dataset_path: Path, max_workers: int = 4) -> None:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        batch_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("full_batch")]

        def index_video(video_dir: Path):
            try:
                return list(streaming_bulk(self.es, self._process_files(video_dir), chunk_size=500, raise_on_error=False))
            except Exception as e:
                print(f"❌ Error indexing {video_dir.name}: {e}")
                return []

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_dir in batch_dirs:
                for video_dir in batch_dir.iterdir():
                    if video_dir.is_dir() and video_dir.name != "maps":
                        futures.append(executor.submit(index_video, video_dir))

            for future in concurrent.futures.as_completed(futures):
                future.result()  # for now, we just wait for all to complete

        print("✅ Finished indexing.")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        body = {
            "query": {
                "more_like_this": {
                    "fields": ["caption"],
                    "like": query,
                    "min_term_freq": 1,
                    "max_query_terms": 12
                }
            }
        }
        response = self.es.search(index=self.index_name, body=body, size=top_k)
        return [SearchResult.from_es_hit(hit) for hit in response['hits']['hits']]
    
    def autocomplete(self, query: str, top_k: int = 3):
        body = {
            "size": top_k,
            "query": {
                "match": {
                    "caption": {
                        "query": query,
                        "operator": "and"
                    }
                }
            }
        }
        response = self.es.search(index=self.index_name, body=body)
        return [SearchResult.from_es_hit(hit) for hit in response['hits']['hits']]

from pathlib import Path

if __name__ == "__main__":
    es_client = ElasticsearchClient()
    # es_client.create_index(recreate=True)
    # es_client.index_dataset(Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset"))

    results = es_client.autocomplete("sit", top_k=5)
    for res in results:
        print(f"[{res.score:.2f}] {res.caption}")
