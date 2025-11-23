#!/usr/bin/env python3
"""
Show the available dictionary data without ML dependencies.

This script demonstrates the data we have available without needing
the sentence-transformers library (which requires PyTorch).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DictionaryLoader, find_common_words, get_word_timeline


def main():
    print("=" * 70)
    print("18th Century Dictionary Data Overview")
    print("=" * 70)

    # Load dictionaries
    loader = DictionaryLoader("data/dictionaries")
    dictionaries = loader.load_all()

    print(f"\nLoaded {len(dictionaries)} dictionaries:\n")
    for d in sorted(dictionaries, key=lambda x: x.year):
        print(f"  📖 {d.name} ({d.year})")
        print(f"     Author: {d.author}")
        print(f"     Entries: {len(d)}")
        print()

    # Find common vocabulary
    common = find_common_words(dictionaries)
    print(f"Common vocabulary across all dictionaries: {len(common)} words")
    print(f"Words: {', '.join(sorted(common))}\n")

    # Show how definitions evolved
    print("=" * 70)
    print("Definition Evolution Examples")
    print("=" * 70)

    interesting_words = ['enthusiasm', 'wit', 'novel', 'patron', 'genius', 'virtue']

    for word in interesting_words:
        if word not in common:
            continue

        print(f"\n📝 {word.upper()}")
        print("-" * 50)

        timeline = get_word_timeline(word, dictionaries)
        for year, entry in timeline:
            # Truncate long definitions
            definition = entry.definition
            if len(definition) > 120:
                definition = definition[:117] + "..."

            print(f"\n  {year} ({entry.source[:20]}):")
            print(f"    \"{definition}\"")

    # Show some famous Johnson definitions
    print("\n" + "=" * 70)
    print("Famous Johnson Definitions")
    print("=" * 70)

    johnson = next((d for d in dictionaries if "Johnson" in d.name), None)
    if johnson:
        famous_words = ['lexicographer', 'oats', 'patron', 'pension', 'excise', 'dull']

        for word in famous_words:
            entry = johnson.get_entry(word)
            if entry:
                print(f"\n  {word.upper()}:")
                print(f"    \"{entry.definition}\"")

    # Show vocabulary differences
    print("\n" + "=" * 70)
    print("Vocabulary Analysis")
    print("=" * 70)

    if len(dictionaries) >= 2:
        sorted_dicts = sorted(dictionaries, key=lambda d: d.year)
        earliest = sorted_dicts[0]
        latest = sorted_dicts[-1]

        words_early = set(earliest.entries.keys())
        words_late = set(latest.entries.keys())

        shared = words_early & words_late
        only_early = words_early - words_late
        only_late = words_late - words_early

        print(f"\n  {earliest.name} ({earliest.year}) vs {latest.name} ({latest.year}):")
        print(f"    Shared words: {len(shared)}")
        print(f"    Only in {earliest.year}: {len(only_early)} - {list(only_early)[:5]}...")
        print(f"    Only in {latest.year}: {len(only_late)} - {list(only_late)[:5]}...")

    print("\n" + "=" * 70)
    print("To run semantic analysis with embeddings, install dependencies:")
    print("  pip install sentence-transformers")
    print("Then run: python examples/quick_demo.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
