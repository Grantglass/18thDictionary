"""Visualization tools for dictionary embeddings and analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Literal

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from .data_loader import Dictionary
from .analysis import SemanticShift


def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["tsne", "pca", "umap"] = "tsne",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        method: Dimensionality reduction method
        n_components: Target dimensions (2 or 3)
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for the reduction method
    """
    if method == "tsne":
        perplexity = kwargs.get("perplexity", min(30, len(embeddings) - 1))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state
        )
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn not installed. Use 'pip install umap-learn'")
        n_neighbors = kwargs.get("n_neighbors", min(15, len(embeddings) - 1))
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)


class DictionaryVisualizer:
    """Visualize dictionary embeddings and analysis results."""

    def __init__(self, output_dir: str | Path = "output/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def plot_embedding_space(
        self,
        embeddings: dict[str, np.ndarray],
        title: str = "Word Embedding Space",
        highlight_words: list[str] | None = None,
        method: str = "tsne",
        save_path: str | None = None,
        show_labels: bool = True,
        max_labels: int = 50
    ) -> plt.Figure:
        """
        Plot 2D projection of word embeddings.

        Args:
            embeddings: Dict mapping words to embeddings
            title: Plot title
            highlight_words: Words to highlight in red
            method: Dimensionality reduction method
            save_path: Path to save figure (relative to output_dir)
            show_labels: Whether to show word labels
            max_labels: Maximum number of labels to show
        """
        words = list(embeddings.keys())
        vectors = np.array([embeddings[w] for w in words])

        # Reduce dimensions
        coords = reduce_dimensions(vectors, method=method)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Determine colors
        highlight_set = set(highlight_words or [])
        colors = ['red' if w in highlight_set else 'steelblue' for w in words]
        sizes = [100 if w in highlight_set else 20 for w in words]

        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, alpha=0.6)

        # Add labels
        if show_labels:
            # Prioritize highlighted words, then sample others
            labeled = set()
            for i, word in enumerate(words):
                if word in highlight_set:
                    ax.annotate(word, (coords[i, 0], coords[i, 1]),
                               fontsize=9, fontweight='bold')
                    labeled.add(word)

            # Add other labels up to max
            remaining = max_labels - len(labeled)
            if remaining > 0:
                indices = np.random.choice(
                    [i for i, w in enumerate(words) if w not in labeled],
                    size=min(remaining, len(words) - len(labeled)),
                    replace=False
                )
                for i in indices:
                    ax.annotate(words[i], (coords[i, 0], coords[i, 1]),
                               fontsize=7, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")

        plt.tight_layout()

        if save_path:
            fig.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_semantic_shifts(
        self,
        shifts: list[SemanticShift],
        title: str = "Semantic Shifts Between Dictionaries",
        top_n: int = 20,
        save_path: str | None = None
    ) -> plt.Figure:
        """
        Plot bar chart of words with largest semantic shifts.
        """
        shifts = sorted(shifts, key=lambda s: s.shift_magnitude, reverse=True)[:top_n]

        fig, ax = plt.subplots(figsize=(12, 8))

        words = [s.word for s in shifts]
        magnitudes = [s.shift_magnitude for s in shifts]

        colors = ['darkred' if m > 0.5 else 'darkorange' if m > 0.3 else 'gold'
                  for m in magnitudes]

        bars = ax.barh(words, magnitudes, color=colors)

        ax.set_xlabel("Semantic Shift Magnitude")
        ax.set_title(title)
        ax.invert_yaxis()

        # Add value labels
        for bar, mag in zip(bars, magnitudes):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{mag:.2f}", va='center', fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_word_evolution(
        self,
        evolution: list[dict],
        word: str,
        save_path: str | None = None
    ) -> plt.Figure:
        """
        Plot how a word's meaning changed over time.
        """
        years = [e["year"] for e in evolution]
        similarities = [e["similarity_to_previous"] for e in evolution]

        # First entry has no previous, so start from second
        years_plot = years[1:]
        sims_plot = [s for s in similarities[1:] if s is not None]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(years_plot, sims_plot, 'o-', markersize=10, linewidth=2)

        # Add horizontal line at threshold
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5,
                   label='Significant change threshold')

        ax.set_xlabel("Year")
        ax.set_ylabel("Similarity to Previous Definition")
        ax.set_title(f"Semantic Evolution of '{word}'")
        ax.set_ylim(0, 1)
        ax.legend()

        # Add source labels
        for i, (year, sim) in enumerate(zip(years_plot, sims_plot)):
            source = evolution[i + 1]["source"]
            ax.annotate(source, (year, sim), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, rotation=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_vocabulary_comparison(
        self,
        dict1: Dictionary,
        dict2: Dictionary,
        save_path: str | None = None
    ) -> plt.Figure:
        """
        Plot Venn-style comparison of vocabulary between dictionaries.
        """
        words1 = set(dict1.entries.keys())
        words2 = set(dict2.entries.keys())

        shared = len(words1 & words2)
        only1 = len(words1 - words2)
        only2 = len(words2 - words1)

        fig, ax = plt.subplots(figsize=(10, 6))

        categories = [f"Only in {dict1.name}\n({dict1.year})",
                     "Shared",
                     f"Only in {dict2.name}\n({dict2.year})"]
        values = [only1, shared, only2]
        colors = ['lightcoral', 'lightgreen', 'lightskyblue']

        bars = ax.bar(categories, values, color=colors, edgecolor='black')

        ax.set_ylabel("Number of Words")
        ax.set_title("Vocabulary Comparison")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   str(val), ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_neighbor_comparison(
        self,
        word: str,
        comparison: dict,
        dict_name_1: str,
        dict_name_2: str,
        save_path: str | None = None
    ) -> plt.Figure:
        """
        Plot comparison of semantic neighborhoods across dictionaries.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: First dictionary neighbors
        neighbors1 = comparison["neighbors_1"]
        words1, sims1 = zip(*neighbors1) if neighbors1 else ([], [])

        ax1 = axes[0]
        colors1 = ['green' if w in comparison["shared"] else 'steelblue' for w in words1]
        ax1.barh(list(words1), list(sims1), color=colors1)
        ax1.set_xlabel("Similarity")
        ax1.set_title(f"Neighbors of '{word}' in {dict_name_1}")
        ax1.invert_yaxis()
        ax1.set_xlim(0, 1)

        # Right: Second dictionary neighbors
        neighbors2 = comparison["neighbors_2"]
        words2, sims2 = zip(*neighbors2) if neighbors2 else ([], [])

        ax2 = axes[1]
        colors2 = ['green' if w in comparison["shared"] else 'steelblue' for w in words2]
        ax2.barh(list(words2), list(sims2), color=colors2)
        ax2.set_xlabel("Similarity")
        ax2.set_title(f"Neighbors of '{word}' in {dict_name_2}")
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1)

        # Add overlap score
        fig.suptitle(f"Neighborhood Overlap: {comparison['overlap_score']:.2%}",
                    fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')

        return fig
