#!/usr/bin/env python3
"""
Example script demonstrating dictionary analysis workflow.

This script shows how to:
1. Load dictionary data
2. Generate embeddings for definitions
3. Analyze semantic shifts between dictionaries
4. Visualize the results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DictionaryLoader, find_common_words
from src.embeddings import EmbeddingGenerator, EmbeddingStore
from src.analysis import DictionaryAnalyzer, compute_vocabulary_overlap
from src.visualization import DictionaryVisualizer


def main():
    # Initialize components
    loader = DictionaryLoader("data/dictionaries")
    generator = EmbeddingGenerator(model_name="default")
    store = EmbeddingStore("output/embeddings")
    visualizer = DictionaryVisualizer("output/figures")

    # Load all dictionaries
    print("Loading dictionaries...")
    dictionaries = loader.load_all()

    if len(dictionaries) < 2:
        print("Need at least 2 dictionaries for comparison.")
        print("Add JSON dictionary files to data/dictionaries/")
        print("\nSee data/dictionaries/example_format.json for the expected format.")
        return

    print(f"Loaded {len(dictionaries)} dictionaries:")
    for d in dictionaries:
        print(f"  - {d.name} ({d.year}): {len(d)} entries")

    # Find common vocabulary
    common_words = find_common_words(dictionaries)
    print(f"\nCommon vocabulary: {len(common_words)} words")

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings_by_dict = {}

    for dictionary in dictionaries:
        cache_name = f"{dictionary.name}_{dictionary.year}"

        if store.exists(cache_name):
            print(f"  Loading cached embeddings for {dictionary.name}...")
            embeddings, _ = store.load(cache_name)
        else:
            print(f"  Generating embeddings for {dictionary.name}...")
            embeddings = generator.embed_dictionary(
                dictionary,
                mode="definition",
                batch_size=32
            )
            store.save(cache_name, embeddings, {
                "name": dictionary.name,
                "year": dictionary.year,
                "model": generator.model_name
            })

        embeddings_by_dict[dictionary.name] = embeddings

    # Initialize analyzer
    analyzer = DictionaryAnalyzer(embeddings_by_dict, dictionaries)

    # Analyze semantic shifts between first and last dictionary
    first_dict = dictionaries[0]
    last_dict = dictionaries[-1]

    print(f"\nAnalyzing semantic shifts: {first_dict.name} -> {last_dict.name}")

    # Find words that changed meaning
    shifts = analyzer.find_semantic_shifts(
        first_dict.name,
        last_dict.name,
        threshold=0.3
    )

    print(f"\nTop 10 words with semantic shift:")
    for shift in shifts[:10]:
        print(f"  {shift.word}: {shift.shift_magnitude:.3f}")
        print(f"    {first_dict.year}: {shift.definition_from[:60]}...")
        print(f"    {last_dict.year}: {shift.definition_to[:60]}...")
        print()

    # Find stable words
    stable = analyzer.find_stable_words(
        first_dict.name,
        last_dict.name,
        threshold=0.1,
        top_k=10
    )

    print(f"Top 10 stable words:")
    for s in stable:
        print(f"  {s.word}: {s.shift_magnitude:.3f}")

    # Visualize results
    print("\nGenerating visualizations...")

    # Plot semantic shifts
    visualizer.plot_semantic_shifts(
        shifts,
        title=f"Semantic Shifts: {first_dict.name} to {last_dict.name}",
        top_n=20,
        save_path="semantic_shifts.png"
    )

    # Plot embedding space for first dictionary
    visualizer.plot_embedding_space(
        embeddings_by_dict[first_dict.name],
        title=f"Word Embedding Space: {first_dict.name} ({first_dict.year})",
        highlight_words=[s.word for s in shifts[:5]],
        save_path=f"embedding_space_{first_dict.year}.png"
    )

    # Vocabulary comparison
    overlap = compute_vocabulary_overlap(first_dict, last_dict)
    print(f"\nVocabulary overlap: {overlap['jaccard_similarity']:.2%}")
    print(f"  Words added: {overlap['only_in_2']}")
    print(f"  Words removed: {overlap['only_in_1']}")

    visualizer.plot_vocabulary_comparison(
        first_dict,
        last_dict,
        save_path="vocabulary_comparison.png"
    )

    # Track evolution of a specific word (if it exists in all dictionaries)
    if common_words:
        sample_word = list(common_words)[0]
        evolution = analyzer.track_word_evolution(sample_word)

        if len(evolution) > 1:
            visualizer.plot_word_evolution(
                evolution,
                sample_word,
                save_path=f"evolution_{sample_word}.png"
            )

    print("\nDone! Check output/figures/ for visualizations.")


if __name__ == "__main__":
    main()
