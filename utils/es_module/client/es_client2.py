import os
import json
from pathlib import Path
from typing import List, Dict, Generator, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import concurrent.futures
from tqdm import tqdm

from ..config.settings import ElasticsearchConfig
from ..models.document import Document, SearchResult

class ElasticsearchClient:
    """Client for interacting with Elasticsearch supporting both OCR and ASR indexing."""

    def __init__(self):
        self.config = ElasticsearchConfig()
        self.es = Elasticsearch(**self.config.get_connection_config())

    def check_connection(self) -> bool:
        return self.es.ping()

    def create_index(self, recreate: bool = False) -> None:
        """Create or recreate index with appropriate mappings (including asr_text and ocr_text)."""
        if recreate and self.es.indices.exists(index=self.config.INDEX_NAME):
            try:
                self.es.indices.delete(
                    index=self.config.INDEX_NAME,
                    request_timeout=120,
                    master_timeout="120s"
                )
                print(f"🧹 Deleted existing index: {self.config.INDEX_NAME}")
            except Exception as e:
                if "process_cluster_event_timeout_exception" in str(e):
                    print(f"⚠️  Timeout while deleting index: {e}")
                    print("💡 Try again after waiting or restart ES.")
                    raise
                else:
                    raise

        if not self.es.indices.exists(index=self.config.INDEX_NAME):
            self.es.indices.create(
                index=self.config.INDEX_NAME,
                body=self.config.get_index_settings()  # mapping must include 'ocr_text' & 'asr_text'
            )
            print(f"Created index: {self.config.INDEX_NAME}")

    def _process_files(self, video_dir: Path) -> Generator[Dict, None, None]:
        """
        Process both OCR and ASR files and yield merged documents for indexing.
        Each frame gets one document, with both ocr_text and asr_text if available.
        """
        ocr_file = video_dir / "ocr_parseq.json"
        asr_file = video_dir / "asr_new.json"
        ocr_data, asr_data = {}, {}

        # if ocr_file.exists():
        try:
            with open(video_dir / "ocr.json", 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
        except:
            with open(video_dir / "ocr_parseq_newmodel.json", 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                
        if asr_file.exists():
            with open(asr_file, 'r', encoding='utf-8') as f:
                asr_data = json.load(f)

        all_frames = set(ocr_data.keys()) | set(asr_data.keys())
        frame_count = 0
        error_frames = {
            'empty_text': [],
            'invalid_text': [],
            'processing_error': []
        }

        for frame_id in all_frames:
            try:
                ocr_text = ocr_data.get(frame_id)
                asr_text = asr_data.get(frame_id)
                # Optionally skip empty docs:
                if (not ocr_text or not ocr_text.strip()) and (not asr_text or not asr_text.strip()):
                    error_frames['empty_text'].append(frame_id)
                    continue

                doc = Document.from_file(video_dir.stem, int(frame_id), ocr_text, asr_text)
                yield doc.to_index_doc(self.config.INDEX_NAME)
                frame_count += 1
            except Exception as e:
                error_frames['processing_error'].append(f"{frame_id} ({str(e)})")

        total_errors = sum(len(errors) for errors in error_frames.values())
        if total_errors > 0:
            print(f"Failed frames ({total_errors}):")
            for k, v in error_frames.items():
                if v:
                    print(f"{k} ({len(v)}): {v[:3]}{' ...' if len(v)>3 else ''}")

    def index_dataset(self, dataset_path: Path) -> None:
        video_list = []
        
        for d in sorted(dataset_path.iterdir()):
            # print(d)
            if d.is_dir() and d.name != "maps":
                video_list.append(d)
        total_videos = len(video_list)

        print(f"Found ({total_videos} videos)\n")

        total_docs = 0
        total_failures: List[str] = []

        # Duyệt tuần tự
        for video_dir in tqdm(video_list):
            batch_success = 0
            batch_failures: List[str] = []

            try:
                for ok, result in streaming_bulk(
                    self.es,
                    self._process_files(video_dir),
                    chunk_size=500,
                    raise_on_error=False
                ):
                    if ok:
                        batch_success += 1
                    else:
                        batch_failures.append(str(result))
            except Exception as e:
                print(f"  Error indexing {video_dir.name}: {e}")
                batch_failures.append(str(e))

            total_docs += batch_success
            total_failures.extend(batch_failures)

            # In gọn kết quả từng video
            if batch_failures:
                print(f"Indexed: {batch_success}, Failures: {len(batch_failures)}")

        # Tổng kết
        print("\nIndexing Summary:")
        print(f"Total videos processed: {total_videos}/{total_videos}")
        print(f"Total frames indexed: {total_docs}")
        if total_failures:
            print(f"Total indexing failures: {len(total_failures)}")
            print("\nFirst 5 failures:")
            for fail in total_failures[:5]:
                print(f"  - {fail}")
            if len(total_failures) > 5:
                print(f"  ... and {len(total_failures) - 5} more failures")
        else:
            print("No failures 🎉")

    def search(
        self,
        query: str,
        size: int = 10,
        fields: Optional[List[str]] = None,
        search_type: str = "combined"
    ) -> List[SearchResult]:
        """
        Search for OCR and/or ASR text.

        Args:
            query: Text to search for
            size: Number of results to return
            fields: Which text fields to search ('ocr_text', 'asr_text', or both)
            search_type: "exact", "fuzzy", or "combined"
        Returns:
            List of SearchResult objects
        """
        if search_type not in ["exact", "fuzzy", "combined"]:
            raise ValueError("search_type must be one of: exact, fuzzy, combined")
        if fields is None:
            fields = ["ocr_text", "asr_text"]

        results = []
        should_clauses_exact = []
        should_clauses_fuzzy = []

        for field in fields:
            should_clauses_exact.append({
                "match_phrase": {
                    field: {
                        "query": query,
                        "slop": 1,
                        "boost": 2.0
                    }
                }
            })
            should_clauses_fuzzy.append({
                "match": {
                    field: {
                        "query": query,
                        "fuzziness": "AUTO",
                        "operator": "or",
                        "boost": 1.0,
                        "minimum_should_match": "60%"
                    }
                }
            })
            should_clauses_fuzzy.append({
                "match_phrase": {
                    field: {
                        "query": query,
                        "slop": 2,
                        "boost": 1.5
                    }
                }
            })

        if search_type in ["exact", "combined"]:
            exact_query = {
                "query": {
                    "bool": {
                        "should": should_clauses_exact,
                        "minimum_should_match": 1
                    }
                },
                "size": size,
                "_source": ["video_id", "frame_id", "ocr_text", "asr_text"],
                "track_scores": True
            }
            exact_hits = self.es.search(
                index=self.config.INDEX_NAME,
                body=exact_query
            )['hits']['hits']
            results.extend(SearchResult.from_es_hit(hit) for hit in exact_hits)

        if search_type in ["fuzzy", "combined"]:
            fuzzy_query = {
                "query": {
                    "bool": {
                        "should": should_clauses_fuzzy,
                        "minimum_should_match": 1
                    }
                },
                "size": size,
                "_source": ["video_id", "frame_id", "ocr_text", "asr_text"],
                "track_scores": True
            }
            fuzzy_hits = self.es.search(
                index=self.config.INDEX_NAME,
                body=fuzzy_query
            )['hits']['hits']

            existing_ids = {r.unique_id for r in results}
            for hit in fuzzy_hits:
                result = SearchResult.from_es_hit(hit)
                if result.unique_id not in existing_ids:
                    results.append(result)
                    existing_ids.add(result.unique_id)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:size]
    
    def get_text_by_frame(self, video_id: str, frame_id: int) -> Optional[Dict[str, str]]:
        """
        Retrieve OCR and ASR text for a specific video_id and frame_id.

        Args:
            video_id: The ID of the video
            frame_id: The ID of the frame

        Returns:
            Dict with 'ocr_text' and 'asr_text', or None if not found
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"video_id": video_id}},
                        {"match": {"frame_id": frame_id}}
                    ]
                }
            },
            "_source": ["ocr_text", "asr_text"]
        }
        
        res = self.es.search(index=self.config.INDEX_NAME, body=query)
        hits = res.get("hits", {}).get("hits", [])

        if hits:
            source = hits[0].get("_source", {})
            return {
                "ocr_text": source.get("ocr_text", ""),
                "asr_text": source.get("asr_text", "")
            }
        return None
