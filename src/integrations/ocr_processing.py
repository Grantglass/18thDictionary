"""
OCR text processing utilities for historical dictionary texts.

This module provides tools for cleaning and parsing OCR output from
scanned historical dictionaries. OCR of 18th century texts often has
characteristic errors that these tools help address.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


# Common OCR errors in 18th century texts
OCR_CORRECTIONS = {
    # Long s (ſ) often misread
    r'ſ': 's',
    r'ﬅ': 'st',
    r'ﬆ': 'st',

    # Common character confusions
    r'vv': 'w',
    r'VV': 'W',
    r'rn': 'm',  # Be careful with this one
    r'cl': 'd',  # Sometimes misread

    # Ligatures
    r'æ': 'ae',
    r'œ': 'oe',
    r'ﬀ': 'ff',
    r'ﬁ': 'fi',
    r'ﬂ': 'fl',
    r'ﬃ': 'ffi',
    r'ﬄ': 'ffl',

    # Quote marks
    r'"': '"',
    r'"': '"',
    r''': "'",
    r''': "'",

    # Common OCR artifacts
    r'\s+': ' ',  # Multiple spaces
    r'- \n': '',  # Hyphenation at line breaks
    r'-\n': '',   # Hyphenation without space
}


@dataclass
class ParsedEntry:
    """A parsed dictionary entry from OCR text."""
    headword: str
    raw_text: str
    definition: str = ""
    part_of_speech: str = ""
    etymology: str = ""
    confidence: float = 1.0  # 0-1, lower means more uncertain parsing


class OCRProcessor:
    """
    Process OCR text from historical dictionaries.

    This class provides methods for:
    1. Cleaning common OCR errors
    2. Splitting text into individual entries
    3. Parsing entry components (headword, definition, etc.)
    """

    def __init__(self):
        self.corrections = OCR_CORRECTIONS.copy()

    def clean_text(self, text: str) -> str:
        """
        Apply standard OCR corrections to text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        result = text

        # Apply character-level corrections
        for pattern, replacement in self.corrections.items():
            if pattern.startswith(r'\s') or pattern.startswith('-'):
                # Regex patterns
                result = re.sub(pattern, replacement, result)
            else:
                # Simple string replacement
                result = result.replace(pattern, replacement)

        # Normalize whitespace
        result = re.sub(r'[ \t]+', ' ', result)
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result.strip()

    def clean_long_s(self, text: str) -> str:
        """
        Convert long s (ſ) to modern s.

        The long s was standard in 18th century printing and is
        often preserved in OCR output.
        """
        # Direct replacement
        text = text.replace('ſ', 's')

        # Sometimes OCR reads long s as f
        # Be conservative here - only fix obvious cases
        # Pattern: f followed by s, t, or end of word in specific contexts
        text = re.sub(r'\bf([st])', r's\1', text)

        return text

    def split_into_entries(
        self,
        text: str,
        pattern: str | None = None
    ) -> list[str]:
        """
        Split continuous text into individual dictionary entries.

        Args:
            text: Full text of dictionary (or section)
            pattern: Optional regex pattern for entry boundaries.
                     Default matches capitalized words at line start.

        Returns:
            List of raw entry texts
        """
        if pattern is None:
            # Default pattern: Entry starts with CAPITALIZED word(s)
            # followed by period or comma
            pattern = r'^([A-Z][A-Z\'\-]+(?:\s+[A-Z]+)*)[.,]'

        # Find all entry starts
        entry_pattern = re.compile(pattern, re.MULTILINE)
        matches = list(entry_pattern.finditer(text))

        if not matches:
            return [text]  # Return whole text as single entry

        entries = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            entry_text = text[start:end].strip()
            if entry_text:
                entries.append(entry_text)

        return entries

    def parse_entry(self, text: str) -> ParsedEntry:
        """
        Parse a single dictionary entry into components.

        Attempts to extract:
        - Headword
        - Part of speech
        - Etymology (if in brackets)
        - Definition

        Args:
            text: Single entry text

        Returns:
            ParsedEntry object
        """
        text = text.strip()
        confidence = 1.0

        # Try to extract headword (first capitalized word(s))
        headword_match = re.match(r'^([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][a-z]*)*)', text)
        if headword_match:
            headword = headword_match.group(1).lower()
            remaining = text[headword_match.end():].strip()
        else:
            # Fallback: first word
            words = text.split()
            headword = words[0].lower().strip('.,;:') if words else ""
            remaining = ' '.join(words[1:]) if len(words) > 1 else ""
            confidence *= 0.7

        # Remove leading punctuation
        remaining = remaining.lstrip('.,;: ')

        # Try to extract part of speech
        pos = ""
        pos_patterns = [
            r'^(n\.s\.|n\. s\.|n\.)',  # noun substantive
            r'^(v\.a\.|v\. a\.)',       # verb active
            r'^(v\.n\.|v\. n\.)',       # verb neuter
            r'^(adj\.|a\.)',            # adjective
            r'^(adv\.)',                # adverb
            r'^(prep\.)',               # preposition
            r'^(conj\.)',               # conjunction
            r'^(interj\.)',             # interjection
            r'^(part\.)',               # participle
        ]

        for pattern in pos_patterns:
            pos_match = re.match(pattern, remaining, re.IGNORECASE)
            if pos_match:
                pos = pos_match.group(1)
                remaining = remaining[pos_match.end():].strip()
                break

        # Try to extract etymology (in square brackets)
        etymology = ""
        etym_match = re.match(r'^\[([^\]]+)\]', remaining)
        if etym_match:
            etymology = etym_match.group(1)
            remaining = remaining[etym_match.end():].strip()

        # What's left is the definition
        definition = remaining.strip('.,;: ')

        # Clean up definition
        definition = re.sub(r'\s+', ' ', definition)

        return ParsedEntry(
            headword=headword,
            raw_text=text,
            definition=definition,
            part_of_speech=pos,
            etymology=etymology,
            confidence=confidence
        )

    def process_dictionary_text(
        self,
        text: str,
        name: str,
        year: int,
        author: str
    ) -> dict:
        """
        Process a full dictionary text into our JSON format.

        Args:
            text: Full OCR text
            name: Dictionary name
            year: Publication year
            author: Author name

        Returns:
            Dictionary in our standard JSON format
        """
        # Clean the text
        cleaned = self.clean_text(text)

        # Split into entries
        entry_texts = self.split_into_entries(cleaned)

        # Parse each entry
        entries = []
        low_confidence = 0

        for entry_text in entry_texts:
            parsed = self.parse_entry(entry_text)

            if parsed.headword and parsed.definition:
                entries.append({
                    "headword": parsed.headword,
                    "definition": parsed.definition,
                    "part_of_speech": parsed.part_of_speech,
                    "etymology": parsed.etymology,
                    "examples": []
                })

                if parsed.confidence < 0.8:
                    low_confidence += 1

        print(f"Parsed {len(entries)} entries ({low_confidence} with low confidence)")

        return {
            "name": name,
            "year": year,
            "author": author,
            "description": f"Processed from OCR text. {low_confidence} entries may need review.",
            "entries": entries
        }


class JohnsonStyleParser(OCRProcessor):
    """
    Specialized parser for Johnson's Dictionary format.

    Johnson's entries typically follow this pattern:
    HEADWORD. part of speech. [Etymology.] Definition text.
    Sometimes followed by numbered senses and quotations.
    """

    def split_into_entries(
        self,
        text: str,
        pattern: str | None = None
    ) -> list[str]:
        """Split using Johnson-style entry pattern."""
        # Johnson entries start with word in caps, then period
        pattern = r'^([A-Z][A-Z\'\-]+)\.'

        return super().split_into_entries(text, pattern)

    def parse_entry(self, text: str) -> ParsedEntry:
        """
        Parse Johnson-style entry.

        Format: HEADWORD. n.s. [Etym.] Definition. Quotation. Author.
        """
        entry = super().parse_entry(text)

        # Johnson often has numbered definitions
        # Try to extract just the first/main one
        definition = entry.definition

        # Remove quotations (usually after definition, attributed to authors)
        # Pattern: text ending with a name or title
        quote_pattern = r'\.\s+[A-Z][a-z]+\.?\s*$'
        definition = re.sub(quote_pattern, '.', definition)

        # If there are numbered senses, take the first
        numbered = re.match(r'^1\.\s*(.+?)(?:\s+2\.|$)', definition, re.DOTALL)
        if numbered:
            definition = numbered.group(1)

        return ParsedEntry(
            headword=entry.headword,
            raw_text=entry.raw_text,
            definition=definition.strip(),
            part_of_speech=entry.part_of_speech,
            etymology=entry.etymology,
            confidence=entry.confidence
        )


class BaileyStyleParser(OCRProcessor):
    """
    Specialized parser for Bailey's Dictionary format.

    Bailey's entries are typically more concise:
    HEADWORD, description or definition.
    """

    def split_into_entries(
        self,
        text: str,
        pattern: str | None = None
    ) -> list[str]:
        """Split using Bailey-style entry pattern."""
        # Bailey entries: WORD, definition (or WORD [origin] definition)
        pattern = r'^([A-Z][A-Z\'\-]+)[,\s]'

        return super().split_into_entries(text, pattern)


def process_ocr_file(
    input_path: str | Path,
    output_path: str | Path,
    name: str,
    year: int,
    author: str,
    style: str = "generic"
) -> None:
    """
    Process an OCR text file into our JSON format.

    Args:
        input_path: Path to OCR text file
        output_path: Path for output JSON
        name: Dictionary name
        year: Publication year
        author: Author name
        style: Parser style - "generic", "johnson", or "bailey"
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Processing {input_path}...")

    text = input_path.read_text(encoding='utf-8', errors='replace')

    if style == "johnson":
        processor = JohnsonStyleParser()
    elif style == "bailey":
        processor = BaileyStyleParser()
    else:
        processor = OCRProcessor()

    result = processor.process_dictionary_text(text, name, year, author)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(result['entries'])} entries to {output_path}")
