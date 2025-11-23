#!/usr/bin/env python3
"""
Build a comparison dataset by extracting common words across dictionaries.

This script:
1. Loads all available dictionary JSON files
2. Finds common vocabulary
3. Creates a focused comparison dataset
4. Adds historical entries from our curated sources
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DictionaryLoader, find_common_words


def load_json(path: Path) -> dict:
    """Load a dictionary JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entries_by_words(dictionary_data: dict, words: set) -> list:
    """Extract entries for specific words from a dictionary."""
    entries = []
    word_to_entry = {e['headword'].lower(): e for e in dictionary_data['entries']}

    for word in words:
        if word.lower() in word_to_entry:
            entries.append(word_to_entry[word.lower()])

    return entries


def main():
    data_dir = Path("data/dictionaries")

    # Load all sample dictionaries
    print("Loading dictionaries...")

    dictionaries = {}

    # Load sample files
    for sample_file in data_dir.glob("*_sample.json"):
        data = load_json(sample_file)
        name = data['name']
        dictionaries[name] = data
        print(f"  Loaded {name}: {len(data['entries'])} entries")

    # Load curated Johnson if available
    curated_path = data_dir / "johnson_1755_curated.json"
    if curated_path.exists():
        data = load_json(curated_path)
        # Merge with sample if exists
        if "Johnson's Dictionary" in dictionaries:
            existing_words = {e['headword'].lower() for e in dictionaries["Johnson's Dictionary"]['entries']}
            for entry in data['entries']:
                if entry['headword'].lower() not in existing_words:
                    dictionaries["Johnson's Dictionary"]['entries'].append(entry)
        else:
            dictionaries[data['name']] = data
        print(f"  Merged curated Johnson entries")

    # Load Webster's if available
    webster_path = data_dir / "webster_1913.json"
    if webster_path.exists():
        data = load_json(webster_path)
        dictionaries[data['name']] = data
        print(f"  Loaded {data['name']}: {len(data['entries'])} entries")

    # Find common vocabulary across historical dictionaries (exclude Webster's for this)
    historical_dicts = {k: v for k, v in dictionaries.items() if '1913' not in k and 'Webster' not in k}

    print(f"\nAnalyzing {len(historical_dicts)} historical dictionaries...")

    # Get all words from each dictionary
    word_sets = []
    for name, data in historical_dicts.items():
        words = {e['headword'].lower() for e in data['entries']}
        word_sets.append(words)
        print(f"  {name}: {len(words)} words")

    # Find common words
    if word_sets:
        common_words = word_sets[0]
        for ws in word_sets[1:]:
            common_words = common_words & ws
        print(f"\nCommon vocabulary: {len(common_words)} words")
        print(f"Words: {sorted(common_words)}")

    # Create expanded historical dataset
    # Add more interesting words for semantic analysis
    interesting_words = {
        # Abstract concepts (often shift meaning)
        'enthusiasm', 'wit', 'genius', 'imagination', 'fancy', 'sensibility',
        'sentiment', 'passion', 'virtue', 'nature', 'art', 'science',

        # Social/political terms
        'liberty', 'justice', 'patron', 'pension', 'excise', 'tory', 'whig',
        'club', 'society', 'government', 'commerce', 'industry',

        # Literary/intellectual terms
        'novel', 'essay', 'critic', 'taste', 'sublime', 'beautiful',
        'dictionary', 'grammar', 'etymology', 'language',

        # Common words that evolved
        'nice', 'awful', 'terrific', 'egregious', 'manufacture',
        'silly', 'cunning', 'brave', 'nervous', 'sad',

        # Technical/scientific terms
        'electricity', 'experiment', 'philosophy', 'mechanic', 'engine',

        # Words with cultural significance
        'gentleman', 'lady', 'honour', 'conversation', 'politeness',
        'coffee', 'tea', 'chocolate', 'fashion', 'luxury'
    }

    # If we have Webster's, extract historical words from it for comparison
    if 'webster_1913.json' in str(webster_path) and webster_path.exists():
        print("\nCreating Webster's subset for comparison...")
        webster_data = load_json(webster_path)

        # Get all words we have in historical dictionaries
        all_historical_words = set()
        for data in historical_dicts.values():
            all_historical_words.update(e['headword'].lower() for e in data['entries'])

        # Also add interesting words
        target_words = all_historical_words | interesting_words

        # Extract matching entries from Webster's
        matching_entries = extract_entries_by_words(webster_data, target_words)

        if matching_entries:
            subset_data = {
                "name": "Webster's Dictionary (Historical Subset)",
                "year": 1913,
                "author": "Noah Webster (revised)",
                "description": f"Subset of {len(matching_entries)} entries matching words from 18th century dictionaries, for semantic comparison.",
                "entries": matching_entries
            }

            output_path = data_dir / "webster_1913_historical_subset.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(subset_data, f, indent=2, ensure_ascii=False)

            print(f"  Created subset with {len(matching_entries)} entries -> {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    for name, data in sorted(dictionaries.items(), key=lambda x: x[1].get('year', 0)):
        year = data.get('year', 'Unknown')
        count = len(data['entries'])
        print(f"  {year}: {name} ({count} entries)")

    print("\nReady for semantic analysis!")
    print("Run: python examples/quick_demo.py")


if __name__ == "__main__":
    main()
