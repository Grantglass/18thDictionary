#!/usr/bin/env python3
"""
Quick demo showing semantic analysis with sample 18th century dictionaries.

This script demonstrates:
1. Loading the sample Bailey (1721) and Johnson (1755) dictionaries
2. Generating embeddings for definitions
3. Finding semantic shifts between them
4. Comparing specific word definitions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DictionaryLoader, find_common_words
from src.embeddings import EmbeddingGenerator, cosine_similarity
from src.analysis import DictionaryAnalyzer


def main():
    print("=" * 60)
    print("18th Century Dictionary Semantic Analysis Demo")
    print("=" * 60)

    # Load sample dictionaries
    loader = DictionaryLoader("data/dictionaries")
    dictionaries = loader.load_all()

    print(f"\nLoaded {len(dictionaries)} dictionaries:")
    for d in sorted(dictionaries, key=lambda x: x.year):
        print(f"  - {d.name} ({d.year}) by {d.author}: {len(d)} entries")

    if len(dictionaries) < 2:
        print("\nNeed at least 2 dictionaries. Check data/dictionaries/")
        return

    # Find common vocabulary
    common = find_common_words(dictionaries)
    print(f"\nCommon words across all dictionaries: {len(common)}")

    # Initialize embedding generator
    print("\nGenerating embeddings (this may take a moment)...")
    generator = EmbeddingGenerator(model_name="default")

    # Generate embeddings for each dictionary
    embeddings = {}
    for d in dictionaries:
        print(f"  Processing {d.name}...")
        embeddings[d.name] = generator.embed_dictionary(d, mode="definition")

    # Create analyzer
    analyzer = DictionaryAnalyzer(embeddings, dictionaries)

    # Sort dictionaries by year
    sorted_dicts = sorted(dictionaries, key=lambda x: x.year)
    first = sorted_dicts[0]
    last = sorted_dicts[-1]

    print(f"\n{'=' * 60}")
    print(f"Comparing: {first.name} ({first.year}) → {last.name} ({last.year})")
    print("=" * 60)

    # Find semantic shifts
    shifts = analyzer.find_semantic_shifts(first.name, last.name, threshold=0.2)

    print(f"\nTop semantic shifts (words that changed meaning):")
    print("-" * 50)
    for i, shift in enumerate(shifts[:10], 1):
        print(f"\n{i}. {shift.word.upper()} (shift: {shift.shift_magnitude:.3f})")
        print(f"   {first.year}: {shift.definition_from[:70]}...")
        print(f"   {last.year}: {shift.definition_to[:70]}...")

    # Find stable words
    stable = analyzer.find_stable_words(first.name, last.name, threshold=0.15, top_k=5)

    print(f"\n\nMost stable words (meaning preserved):")
    print("-" * 50)
    for shift in stable:
        print(f"  {shift.word}: {shift.shift_magnitude:.3f}")

    # Demonstrate specific word analysis
    print(f"\n{'=' * 60}")
    print("Detailed Word Analysis")
    print("=" * 60)

    interesting_words = ["enthusiasm", "wit", "novel", "patron"]

    for word in interesting_words:
        if word not in common:
            continue

        print(f"\n{word.upper()}")
        print("-" * 40)

        # Show evolution
        evolution = analyzer.track_word_evolution(word)
        for entry in evolution:
            print(f"  {entry['year']} ({entry['source'][:15]}):")
            print(f"    \"{entry['definition'][:80]}...\"")
            if entry['similarity_to_previous'] is not None:
                sim = entry['similarity_to_previous']
                change = "stable" if sim > 0.8 else "shifted" if sim < 0.6 else "some change"
                print(f"    → Similarity to previous: {sim:.3f} ({change})")

    # Show semantic neighbors
    print(f"\n{'=' * 60}")
    print("Semantic Neighborhoods")
    print("=" * 60)

    sample_word = "virtue"
    if sample_word in common:
        print(f"\nWords semantically similar to '{sample_word}':")

        for d in sorted_dicts:
            neighbors = analyzer.get_word_neighbors(sample_word, d.name, top_k=5)
            if neighbors:
                print(f"\n  {d.name} ({d.year}):")
                for word, sim in neighbors:
                    print(f"    - {word}: {sim:.3f}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
