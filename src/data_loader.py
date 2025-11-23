"""Data loading utilities for 18th century dictionary texts."""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class DictionaryEntry:
    """Represents a single dictionary entry."""

    headword: str
    definition: str
    part_of_speech: str = ""
    etymology: str = ""
    examples: list[str] = field(default_factory=list)
    source: str = ""
    year: int = 0

    def __str__(self) -> str:
        return f"{self.headword}: {self.definition[:100]}..."


@dataclass
class Dictionary:
    """Represents a complete dictionary with metadata."""

    name: str
    year: int
    author: str
    entries: dict[str, DictionaryEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[DictionaryEntry]:
        return iter(self.entries.values())

    def get_entry(self, word: str) -> DictionaryEntry | None:
        """Get entry by headword (case-insensitive)."""
        return self.entries.get(word.lower())

    def get_words(self) -> list[str]:
        """Get all headwords."""
        return list(self.entries.keys())


class DictionaryLoader:
    """Load dictionary data from various formats."""

    def __init__(self, data_dir: str | Path = "data/dictionaries"):
        self.data_dir = Path(data_dir)

    def load_json(self, filepath: str | Path) -> Dictionary:
        """
        Load dictionary from JSON format.

        Expected format:
        {
            "name": "Johnson's Dictionary",
            "year": 1755,
            "author": "Samuel Johnson",
            "entries": [
                {
                    "headword": "word",
                    "definition": "...",
                    "part_of_speech": "n.",
                    "etymology": "...",
                    "examples": ["..."]
                }
            ]
        }
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dictionary = Dictionary(
            name=data.get("name", "Unknown"),
            year=data.get("year", 0),
            author=data.get("author", "Unknown")
        )

        for entry_data in data.get("entries", []):
            entry = DictionaryEntry(
                headword=entry_data["headword"].lower(),
                definition=entry_data.get("definition", ""),
                part_of_speech=entry_data.get("part_of_speech", ""),
                etymology=entry_data.get("etymology", ""),
                examples=entry_data.get("examples", []),
                source=dictionary.name,
                year=dictionary.year
            )
            dictionary.entries[entry.headword] = entry

        return dictionary

    def load_plain_text(
        self,
        filepath: str | Path,
        name: str,
        year: int,
        author: str,
        entry_pattern: str = r"^([A-Z][A-Za-z]*)\.\s+(.+)$"
    ) -> Dictionary:
        """
        Load dictionary from plain text with configurable pattern.

        Default pattern matches: "Word. definition text"
        """
        dictionary = Dictionary(name=name, year=year, author=author)
        pattern = re.compile(entry_pattern, re.MULTILINE)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        for match in pattern.finditer(content):
            headword = match.group(1).lower()
            definition = match.group(2).strip()

            entry = DictionaryEntry(
                headword=headword,
                definition=definition,
                source=name,
                year=year
            )
            dictionary.entries[headword] = entry

        return dictionary

    def load_all(self) -> list[Dictionary]:
        """Load all dictionaries from the data directory."""
        dictionaries = []

        for json_file in self.data_dir.glob("*.json"):
            try:
                dictionaries.append(self.load_json(json_file))
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return sorted(dictionaries, key=lambda d: d.year)


def find_common_words(dictionaries: list[Dictionary]) -> set[str]:
    """Find words that appear in all dictionaries."""
    if not dictionaries:
        return set()

    common = set(dictionaries[0].entries.keys())
    for dictionary in dictionaries[1:]:
        common &= set(dictionary.entries.keys())

    return common


def get_word_timeline(word: str, dictionaries: list[Dictionary]) -> list[tuple[int, DictionaryEntry]]:
    """Get all definitions of a word across dictionaries, sorted by year."""
    timeline = []
    word = word.lower()

    for dictionary in dictionaries:
        entry = dictionary.get_entry(word)
        if entry:
            timeline.append((dictionary.year, entry))

    return sorted(timeline, key=lambda x: x[0])
