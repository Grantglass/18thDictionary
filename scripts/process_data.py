#!/usr/bin/env python3
"""
Data processing scripts for converting dictionary data into our JSON format.

This module provides utilities for:
1. Converting Webster's dictionary (JSON) to our format
2. Processing plain text OCR output from historical dictionaries
3. Parsing common historical dictionary formats
"""

import json
import re
import sys
from pathlib import Path


def convert_webster_dict(input_path: str, output_path: str) -> None:
    """
    Convert the adambom/dictionary format to our standard format.

    Input format: {"WORD": "Definition", ...}
    Output format: Our standard JSON with metadata and entry objects

    Usage:
        python process_data.py convert-webster /path/to/dictionary.json output.json
    """
    print(f"Loading Webster's dictionary from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    entries = []
    for word, definition in raw_data.items():
        # Clean up the definition
        definition = definition.strip()
        definition = re.sub(r'\s+', ' ', definition)

        # Extract part of speech if present at the start
        pos_match = re.match(r'^([a-z]\.\s*(?:[a-z]\.)?\s*)', definition, re.IGNORECASE)
        pos = ""
        if pos_match:
            pos = pos_match.group(1).strip()
            definition = definition[len(pos_match.group(0)):].strip()

        entries.append({
            "headword": word.lower(),
            "definition": definition,
            "part_of_speech": pos,
            "etymology": "",
            "examples": []
        })

    output_data = {
        "name": "Webster's Unabridged Dictionary",
        "year": 1913,
        "author": "Noah Webster (revised)",
        "description": "Webster's Unabridged Dictionary, converted from Project Gutenberg text.",
        "entries": entries
    }

    print(f"Writing {len(entries)} entries to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Done!")


def parse_johnson_format(text: str) -> list[dict]:
    """
    Parse text in Johnson's dictionary format.

    Format:
    HEADWORD. part of speech. [Etymology.] Definition text.
    """
    entries = []

    # Pattern for Johnson-style entries
    # Matches: WORD. n.s. [etym.] Definition...
    pattern = re.compile(
        r'^([A-Z][A-Za-z\'\-]+)\.\s*'  # Headword
        r'(?:([a-z]\.\s*(?:s\.)?)\s*)?'  # Part of speech (optional)
        r'(?:\[([^\]]+)\]\s*)?'  # Etymology in brackets (optional)
        r'(.+?)(?=^[A-Z][A-Za-z\'\-]+\.|$)',  # Definition until next entry
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(text):
        headword = match.group(1).lower()
        pos = match.group(2) or ""
        etymology = match.group(3) or ""
        definition = match.group(4).strip()

        # Clean up definition
        definition = re.sub(r'\s+', ' ', definition)

        entries.append({
            "headword": headword,
            "definition": definition,
            "part_of_speech": pos.strip(),
            "etymology": etymology.strip(),
            "examples": []
        })

    return entries


def parse_simple_format(text: str) -> list[dict]:
    """
    Parse simple "Word: Definition" format.

    Format:
    Word: Definition text
    or
    Word - Definition text
    """
    entries = []

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try colon separator
        if ':' in line:
            parts = line.split(':', 1)
        elif ' - ' in line:
            parts = line.split(' - ', 1)
        else:
            continue

        if len(parts) == 2:
            headword = parts[0].strip().lower()
            definition = parts[1].strip()

            if headword and definition:
                entries.append({
                    "headword": headword,
                    "definition": definition,
                    "part_of_speech": "",
                    "etymology": "",
                    "examples": []
                })

    return entries


def process_ocr_text(
    input_path: str,
    output_path: str,
    name: str,
    year: int,
    author: str,
    format_type: str = "simple"
) -> None:
    """
    Process OCR text file and convert to our JSON format.

    Args:
        input_path: Path to plain text file
        output_path: Path for output JSON
        name: Dictionary name
        year: Publication year
        author: Author name
        format_type: "simple" for Word: Definition, "johnson" for historical format
    """
    print(f"Processing {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if format_type == "johnson":
        entries = parse_johnson_format(text)
    else:
        entries = parse_simple_format(text)

    output_data = {
        "name": name,
        "year": year,
        "author": author,
        "description": f"Converted from OCR text.",
        "entries": entries
    }

    print(f"Found {len(entries)} entries")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


def merge_dictionaries(input_paths: list[str], output_path: str) -> None:
    """
    Merge multiple dictionary JSON files, keeping unique entries.
    """
    all_entries = {}
    metadata = None

    for path in input_paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if metadata is None:
            metadata = {
                "name": data.get("name", "Merged Dictionary"),
                "year": data.get("year", 0),
                "author": data.get("author", "Various"),
                "description": "Merged from multiple sources"
            }

        for entry in data.get("entries", []):
            word = entry["headword"].lower()
            if word not in all_entries:
                all_entries[word] = entry
            else:
                # Merge: prefer longer definitions
                if len(entry["definition"]) > len(all_entries[word]["definition"]):
                    all_entries[word] = entry

    output_data = {
        **metadata,
        "entries": list(all_entries.values())
    }

    print(f"Merged {len(all_entries)} unique entries")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


def validate_dictionary(path: str) -> bool:
    """
    Validate a dictionary JSON file.
    """
    print(f"Validating {path}...")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON - {e}")
        return False

    required_fields = ["name", "year", "author", "entries"]
    for field in required_fields:
        if field not in data:
            print(f"  ERROR: Missing required field '{field}'")
            return False

    entry_fields = ["headword", "definition"]
    errors = 0

    for i, entry in enumerate(data["entries"]):
        for field in entry_fields:
            if field not in entry:
                print(f"  ERROR: Entry {i} missing '{field}'")
                errors += 1
            elif not entry[field]:
                print(f"  WARNING: Entry {i} has empty '{field}'")

    if errors > 0:
        print(f"  Found {errors} errors")
        return False

    print(f"  Valid: {len(data['entries'])} entries")
    return True


def print_usage():
    """Print usage information."""
    print("""
Dictionary Data Processing Tools

Usage:
    python process_data.py <command> [arguments]

Commands:
    convert-webster <input.json> <output.json>
        Convert Webster's dictionary format to our standard format

    process-ocr <input.txt> <output.json> <name> <year> <author> [format]
        Process OCR text file. Format can be 'simple' or 'johnson'

    merge <output.json> <input1.json> [input2.json ...]
        Merge multiple dictionary files

    validate <dictionary.json>
        Validate a dictionary file

Examples:
    python process_data.py convert-webster webster_raw.json webster.json
    python process_data.py process-ocr johnson.txt johnson.json "Johnson's Dictionary" 1755 "Samuel Johnson" johnson
    python process_data.py validate data/dictionaries/johnson_1755_sample.json
    """)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    if command == "convert-webster" and len(sys.argv) == 4:
        convert_webster_dict(sys.argv[2], sys.argv[3])

    elif command == "process-ocr" and len(sys.argv) >= 7:
        format_type = sys.argv[7] if len(sys.argv) > 7 else "simple"
        process_ocr_text(
            sys.argv[2], sys.argv[3], sys.argv[4],
            int(sys.argv[5]), sys.argv[6], format_type
        )

    elif command == "merge" and len(sys.argv) >= 4:
        merge_dictionaries(sys.argv[3:], sys.argv[2])

    elif command == "validate" and len(sys.argv) == 3:
        valid = validate_dictionary(sys.argv[2])
        sys.exit(0 if valid else 1)

    else:
        print_usage()


if __name__ == "__main__":
    main()
