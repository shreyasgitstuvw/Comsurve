"""
Qdrant local-mode vector store manager.

One collection per commodity (lng, copper, soybeans).
Collection name format: mcei_{commodity}

Each vector point payload:
  {
    "anomaly_event_id": int,
    "anomaly_type":     str,
    "commodity":        str,
    "severity":         float,
    "detected_at":      str (ISO),
    "context_snippet":  str (first 300 chars of context payload)
  }

Vector dimension: 3072 (gemini-embedding-001).
Distance metric: Cosine (appropriate for semantic text embeddings).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    QueryResponse,
    VectorParams,
)

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

VECTOR_SIZE = 3072
COLLECTION_PREFIX = "mcei"


class QdrantManager:
    """
    Manages local Qdrant collections for MCEI.
    One instance per process — reuse across all ai_engine operations.
    """

    def __init__(self):
        self._client = QdrantClient(path=settings.qdrant_path)
        self._closed = False
        self._ensure_collections()

    def _collection_name(self, commodity: str) -> str:
        return f"{COLLECTION_PREFIX}_{commodity}"

    def _ensure_collections(self) -> None:
        """Create commodity collections if they don't already exist."""
        from shared.commodity_registry import COMMODITY_LIST
        existing = {c.name for c in self._client.get_collections().collections}

        for commodity in COMMODITY_LIST:
            name = self._collection_name(commodity)
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                logger.info("qdrant_collection_created", collection=name)

    def upsert_embedding(
        self,
        anomaly_event_id: int,
        commodity: str,
        vector: list[float],
        payload: dict,
    ) -> None:
        """
        Insert or update a vector point for an anomaly event.
        Point ID = anomaly_event_id (integer, stable across reruns).
        """
        collection = self._collection_name(commodity)
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=anomaly_event_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.info("qdrant_upsert", collection=collection, point_id=anomaly_event_id)

    def search_similar(
        self,
        commodity: str,
        vector: list[float],
        top_k: int = 5,
        exclude_id: int | None = None,
        min_score: float = 0.70,
    ) -> list[QueryResponse]:
        """
        Find the top_k most similar historical anomalies for this commodity.

        exclude_id: the current anomaly's own ID — excluded from results.
        min_score:  cosine similarity floor (0.70 = reasonably similar).
        """
        collection = self._collection_name(commodity)
        fetch_limit = top_k + (1 if exclude_id is not None else 0)

        results: list[QueryResponse] = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=fetch_limit,
            score_threshold=min_score,
            with_payload=True,
        ).points

        if exclude_id is not None:
            results = [r for r in results if r.id != exclude_id]

        return results[:top_k]

    def count(self, commodity: str) -> int:
        """Return number of vectors stored for a commodity."""
        collection = self._collection_name(commodity)
        return self._client.count(collection_name=collection).count

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
