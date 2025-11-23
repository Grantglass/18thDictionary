"""Word embedding generation and management for dictionary analysis."""

import numpy as np
from pathlib import Path
from typing import Literal

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .data_loader import Dictionary, DictionaryEntry


class EmbeddingGenerator:
    """Generate embeddings for dictionary entries using transformer models."""

    # Models suitable for historical/semantic analysis
    MODELS = {
        "default": "all-MiniLM-L6-v2",  # Fast, good quality
        "accurate": "all-mpnet-base-v2",  # More accurate, slower
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # For non-English
    }

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Model to use (key from MODELS or HuggingFace model name)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        model_name = model_name or self.MODELS["default"]
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def embed_entry(
        self,
        entry: DictionaryEntry,
        mode: Literal["definition", "headword", "combined"] = "definition"
    ) -> np.ndarray:
        """
        Generate embedding for a dictionary entry.

        Args:
            entry: Dictionary entry to embed
            mode: What to embed:
                - "definition": Just the definition text
                - "headword": Just the headword
                - "combined": "headword: definition"
        """
        if mode == "definition":
            text = entry.definition
        elif mode == "headword":
            text = entry.headword
        else:  # combined
            text = f"{entry.headword}: {entry.definition}"

        return self.embed_text(text)

    def embed_dictionary(
        self,
        dictionary: Dictionary,
        mode: Literal["definition", "headword", "combined"] = "definition",
        batch_size: int = 32,
        show_progress: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings for all entries in a dictionary.

        Returns:
            Dictionary mapping headwords to their embeddings
        """
        entries = list(dictionary.entries.values())
        headwords = [e.headword for e in entries]

        if mode == "definition":
            texts = [e.definition for e in entries]
        elif mode == "headword":
            texts = headwords
        else:
            texts = [f"{e.headword}: {e.definition}" for e in entries]

        embeddings = self.embed_texts(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        return dict(zip(headwords, embeddings))


class EmbeddingStore:
    """Store and retrieve embeddings efficiently."""

    def __init__(self, cache_dir: str | Path = "output/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        """Get path for cached embeddings."""
        return self.cache_dir / f"{name}.npz"

    def save(
        self,
        name: str,
        embeddings: dict[str, np.ndarray],
        metadata: dict | None = None
    ) -> None:
        """Save embeddings to disk."""
        words = list(embeddings.keys())
        vectors = np.array([embeddings[w] for w in words])

        save_dict = {
            "words": np.array(words, dtype=object),
            "vectors": vectors
        }

        if metadata:
            save_dict["metadata"] = np.array([metadata], dtype=object)

        np.savez_compressed(self._get_cache_path(name), **save_dict)

    def load(self, name: str) -> tuple[dict[str, np.ndarray], dict | None]:
        """Load embeddings from disk."""
        data = np.load(self._get_cache_path(name), allow_pickle=True)

        words = data["words"].tolist()
        vectors = data["vectors"]
        embeddings = dict(zip(words, vectors))

        metadata = None
        if "metadata" in data:
            metadata = data["metadata"][0]

        return embeddings, metadata

    def exists(self, name: str) -> bool:
        """Check if cached embeddings exist."""
        return self._get_cache_path(name).exists()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_nearest_neighbors(
    query: np.ndarray,
    embeddings: dict[str, np.ndarray],
    top_k: int = 10,
    exclude: set[str] | None = None
) -> list[tuple[str, float]]:
    """
    Find nearest neighbors to a query embedding.

    Returns:
        List of (word, similarity) tuples, sorted by similarity descending
    """
    exclude = exclude or set()
    similarities = []

    for word, embedding in embeddings.items():
        if word in exclude:
            continue
        sim = cosine_similarity(query, embedding)
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
