"""Analysis tools for comparing dictionary entries across time."""

import numpy as np
from dataclasses import dataclass
from typing import Literal

from .data_loader import Dictionary, DictionaryEntry, get_word_timeline
from .embeddings import cosine_similarity, find_nearest_neighbors


@dataclass
class SemanticShift:
    """Represents a semantic shift for a word between two time periods."""

    word: str
    year_from: int
    year_to: int
    source_from: str
    source_to: str
    definition_from: str
    definition_to: str
    similarity: float
    shift_magnitude: float  # 1 - similarity, higher = more change

    @property
    def changed_significantly(self) -> bool:
        """Whether the word underwent significant semantic change."""
        return self.shift_magnitude > 0.3


@dataclass
class WordCluster:
    """A cluster of semantically related words."""

    center_word: str
    members: list[tuple[str, float]]  # (word, similarity to center)
    source: str
    year: int


class DictionaryAnalyzer:
    """Analyze and compare dictionary embeddings."""

    def __init__(
        self,
        embeddings_by_dictionary: dict[str, dict[str, np.ndarray]],
        dictionaries: list[Dictionary]
    ):
        """
        Initialize analyzer.

        Args:
            embeddings_by_dictionary: Map of dictionary name to word->embedding dict
            dictionaries: List of Dictionary objects with metadata
        """
        self.embeddings = embeddings_by_dictionary
        self.dictionaries = {d.name: d for d in dictionaries}
        self.dict_by_year = {d.year: d for d in dictionaries}

    def compare_definitions(
        self,
        word: str,
        dict_name_1: str,
        dict_name_2: str
    ) -> SemanticShift | None:
        """
        Compare embeddings of a word between two dictionaries.

        Returns:
            SemanticShift object or None if word not in both dictionaries
        """
        word = word.lower()

        emb1 = self.embeddings.get(dict_name_1, {}).get(word)
        emb2 = self.embeddings.get(dict_name_2, {}).get(word)

        if emb1 is None or emb2 is None:
            return None

        dict1 = self.dictionaries[dict_name_1]
        dict2 = self.dictionaries[dict_name_2]

        sim = cosine_similarity(emb1, emb2)

        return SemanticShift(
            word=word,
            year_from=dict1.year,
            year_to=dict2.year,
            source_from=dict_name_1,
            source_to=dict_name_2,
            definition_from=dict1.entries[word].definition,
            definition_to=dict2.entries[word].definition,
            similarity=sim,
            shift_magnitude=1 - sim
        )

    def find_semantic_shifts(
        self,
        dict_name_1: str,
        dict_name_2: str,
        threshold: float = 0.3,
        min_words: int = 10
    ) -> list[SemanticShift]:
        """
        Find words that changed meaning between two dictionaries.

        Args:
            dict_name_1: Name of first (earlier) dictionary
            dict_name_2: Name of second (later) dictionary
            threshold: Minimum shift magnitude to include
            min_words: Return at least this many words even if below threshold

        Returns:
            List of SemanticShift objects, sorted by shift magnitude
        """
        emb1 = self.embeddings.get(dict_name_1, {})
        emb2 = self.embeddings.get(dict_name_2, {})

        # Find common words
        common_words = set(emb1.keys()) & set(emb2.keys())

        shifts = []
        for word in common_words:
            shift = self.compare_definitions(word, dict_name_1, dict_name_2)
            if shift:
                shifts.append(shift)

        # Sort by shift magnitude (descending)
        shifts.sort(key=lambda s: s.shift_magnitude, reverse=True)

        # Apply threshold but ensure minimum results
        filtered = [s for s in shifts if s.shift_magnitude >= threshold]
        if len(filtered) < min_words:
            return shifts[:min_words]
        return filtered

    def find_stable_words(
        self,
        dict_name_1: str,
        dict_name_2: str,
        threshold: float = 0.1,
        top_k: int = 50
    ) -> list[SemanticShift]:
        """
        Find words that maintained their meaning between dictionaries.

        Args:
            threshold: Maximum shift magnitude to consider "stable"
            top_k: Number of results to return
        """
        emb1 = self.embeddings.get(dict_name_1, {})
        emb2 = self.embeddings.get(dict_name_2, {})

        common_words = set(emb1.keys()) & set(emb2.keys())

        shifts = []
        for word in common_words:
            shift = self.compare_definitions(word, dict_name_1, dict_name_2)
            if shift and shift.shift_magnitude <= threshold:
                shifts.append(shift)

        shifts.sort(key=lambda s: s.shift_magnitude)
        return shifts[:top_k]

    def get_word_neighbors(
        self,
        word: str,
        dict_name: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find words with similar definitions in a dictionary."""
        word = word.lower()
        embeddings = self.embeddings.get(dict_name, {})

        if word not in embeddings:
            return []

        query = embeddings[word]
        return find_nearest_neighbors(
            query, embeddings, top_k=top_k + 1, exclude={word}
        )[:top_k]

    def compare_neighborhoods(
        self,
        word: str,
        dict_name_1: str,
        dict_name_2: str,
        top_k: int = 10
    ) -> dict:
        """
        Compare semantic neighborhoods of a word across dictionaries.

        Returns dict with:
            - neighbors_1: neighbors in first dictionary
            - neighbors_2: neighbors in second dictionary
            - shared: words in both neighborhoods
            - only_in_1: words only in first neighborhood
            - only_in_2: words only in second neighborhood
            - overlap_score: Jaccard similarity of neighborhoods
        """
        n1 = self.get_word_neighbors(word, dict_name_1, top_k)
        n2 = self.get_word_neighbors(word, dict_name_2, top_k)

        words1 = set(w for w, _ in n1)
        words2 = set(w for w, _ in n2)

        shared = words1 & words2
        overlap = len(shared) / len(words1 | words2) if words1 | words2 else 0

        return {
            "neighbors_1": n1,
            "neighbors_2": n2,
            "shared": shared,
            "only_in_1": words1 - words2,
            "only_in_2": words2 - words1,
            "overlap_score": overlap
        }

    def track_word_evolution(
        self,
        word: str
    ) -> list[dict]:
        """
        Track how a word's meaning evolved across all dictionaries.

        Returns list of dicts with year, source, definition, and
        similarity to previous definition.
        """
        word = word.lower()
        evolution = []
        prev_embedding = None

        # Sort dictionaries by year
        sorted_dicts = sorted(self.dictionaries.values(), key=lambda d: d.year)

        for dictionary in sorted_dicts:
            if word not in self.embeddings.get(dictionary.name, {}):
                continue

            entry = dictionary.entries.get(word)
            embedding = self.embeddings[dictionary.name][word]

            similarity_to_prev = None
            if prev_embedding is not None:
                similarity_to_prev = cosine_similarity(prev_embedding, embedding)

            evolution.append({
                "year": dictionary.year,
                "source": dictionary.name,
                "definition": entry.definition if entry else "",
                "similarity_to_previous": similarity_to_prev
            })

            prev_embedding = embedding

        return evolution


def compute_vocabulary_overlap(dict1: Dictionary, dict2: Dictionary) -> dict:
    """
    Compute vocabulary overlap statistics between dictionaries.

    Returns:
        Dict with overlap metrics
    """
    words1 = set(dict1.entries.keys())
    words2 = set(dict2.entries.keys())

    shared = words1 & words2
    only_in_1 = words1 - words2
    only_in_2 = words2 - words1

    return {
        "total_1": len(words1),
        "total_2": len(words2),
        "shared": len(shared),
        "only_in_1": len(only_in_1),
        "only_in_2": len(only_in_2),
        "jaccard_similarity": len(shared) / len(words1 | words2),
        "words_added": only_in_2,
        "words_removed": only_in_1
    }
