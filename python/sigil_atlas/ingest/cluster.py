"""KMeans clustering at multiple granularities, precomputed at ingest time.

For each embedding model, runs MiniBatchKMeans at several k values and stores
assignments + centroids in the DB. The layout reads these at serve time —
tightness slider selects which k level to use.

Resumable: skips k levels already computed.
"""

import logging
import struct

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)

KMEANS_K_LEVELS = [10, 25, 50, 100, 200, 500, 1000]


def _pack_vector(v: np.ndarray) -> bytes:
    return struct.pack(f"<{len(v)}f", *v.tolist())


def run_clustering_stage(
    db: CorpusDB,
    provider: EmbeddingProvider,
    model: str,
    k_levels: list[int] | None = None,
    progress: StageProgress | None = None,
    token: CancellationToken | None = None,
) -> None:
    if k_levels is None:
        k_levels = KMEANS_K_LEVELS

    embedded_ids = sorted(db.fetch_embedded_image_ids(model))
    n = len(embedded_ids)
    if n < 2:
        logger.info("Clustering: fewer than 2 images for %s, skipping", model)
        return

    # Filter k levels to those not already computed and feasible
    pending = [k for k in k_levels if k <= n and not db.has_kmeans(model, k)]
    if not pending:
        logger.info("Clustering: all k levels cached for %s", model)
        if progress:
            progress.advance(len(k_levels))
        return

    logger.info("Clustering: loading %d embeddings for %s", n, model)
    matrix = provider.fetch_matrix(embedded_ids, model)

    for k in pending:
        if token and token.is_cancelled:
            return

        logger.info("Clustering: MiniBatchKMeans k=%d on %d images (%s)", k, n, model)
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(1024, n))
        labels = kmeans.fit_predict(matrix)

        assignments = [(embedded_ids[i], int(labels[i])) for i in range(n)]
        centroids = {i: _pack_vector(kmeans.cluster_centers_[i]) for i in range(k)}

        db.insert_kmeans_batch(model, k, assignments)
        db.insert_kmeans_centroids(model, k, centroids)
        logger.info("Clustering: k=%d done for %s", k, model)

        if progress:
            progress.advance(1)
