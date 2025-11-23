#!/usr/bin/env python3
"""
Unified CLI for fetching dictionary data from various sources.

Usage:
    python scripts/fetch_data.py <command> [options]

Commands:
    list-sources          List all available data sources
    list-dictionaries     List known dictionaries across all sources
    fetch-archive         Download from Internet Archive
    fetch-hathi          Get info/instructions for HathiTrust
    export-johnson       Export curated Johnson entries
    process-ocr          Process downloaded OCR text

Examples:
    python scripts/fetch_data.py list-dictionaries
    python scripts/fetch_data.py fetch-archive johnson_1755_v1
    python scripts/fetch_data.py export-johnson data/dictionaries/johnson_curated.json
    python scripts/fetch_data.py process-ocr downloaded.txt output.json "Dictionary Name" 1755 "Author" johnson
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.internet_archive import InternetArchiveClient
from src.integrations.hathitrust import HathiTrustClient
from src.integrations.johnson_online import JohnsonDictionaryClient
from src.integrations.ocr_processing import process_ocr_file


def list_sources():
    """List all available data sources."""
    print("""
Available Data Sources
======================

1. Internet Archive (archive.org)
   - Free access to scanned historical texts
   - OCR text available for many items
   - Use: fetch-archive command

2. HathiTrust Digital Library
   - Large collection of digitized books
   - Some require institutional access
   - Use: fetch-hathi command for info

3. Johnson's Dictionary Online (johnsonsdictionaryonline.com)
   - Searchable interface to Johnson's Dictionary
   - Curated entries available offline
   - Use: export-johnson command

4. Local OCR Processing
   - Process your own downloaded/OCR'd texts
   - Supports multiple dictionary styles
   - Use: process-ocr command
""")


def list_dictionaries():
    """List all known dictionaries from all sources."""
    print("\n" + "=" * 70)
    print("Known 18th Century Dictionaries")
    print("=" * 70)

    # Internet Archive
    print("\n📚 Internet Archive")
    print("-" * 50)
    archive = InternetArchiveClient()
    for key, identifier in archive.list_known_dictionaries().items():
        print(f"  {key:25} -> {identifier}")

    # HathiTrust
    print("\n📖 HathiTrust")
    print("-" * 50)
    hathi = HathiTrustClient()
    for key, htid in hathi.list_known_dictionaries().items():
        print(f"  {key:25} -> {htid}")

    # Johnson's curated
    print("\n✍️  Johnson's Dictionary (Curated Entries)")
    print("-" * 50)
    johnson = JohnsonDictionaryClient()
    words = johnson.list_available_words()
    print(f"  {len(words)} curated entries available")
    print(f"  Words: {', '.join(words[:10])}...")

    print("\n" + "=" * 70)
    print("Use 'fetch-archive <key>' or 'fetch-hathi <key>' to get data")
    print("=" * 70)


def fetch_archive(key: str, output_dir: str = "data/raw"):
    """Download dictionary text from Internet Archive."""
    client = InternetArchiveClient()

    known = client.list_known_dictionaries()
    if key not in known:
        print(f"Unknown key: {key}")
        print(f"Available: {list(known.keys())}")
        return

    print(f"\nFetching {key} from Internet Archive...")

    # Get item info first
    identifier = known[key]
    item = client.get_item(identifier)

    if item:
        print(f"  Title: {item.title}")
        print(f"  Author: {item.creator}")
        print(f"  Year: {item.year}")
        print(f"  Formats: {', '.join(item.formats[:5])}")
        print(f"  URL: {item.url}")

    # Try to download text
    text = client.download_known_dictionary(key)

    if text:
        # Save raw text
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        raw_file = output_path / f"{key}_raw.txt"
        raw_file.write_text(text, encoding='utf-8')
        print(f"\n✓ Saved raw text to {raw_file}")
        print(f"  Size: {len(text):,} characters")

        print(f"\nNext steps:")
        print(f"  1. Review the raw text in {raw_file}")
        print(f"  2. Process with: python scripts/fetch_data.py process-ocr {raw_file} data/dictionaries/{key}.json \"Name\" YEAR \"Author\" johnson")
    else:
        print("\n✗ Could not download text file")
        print(f"  Try visiting: {item.url if item else f'https://archive.org/details/{identifier}'}")


def fetch_hathi(key: str):
    """Get information and download instructions for HathiTrust."""
    client = HathiTrustClient()

    known = client.list_known_dictionaries()
    if key not in known:
        print(f"Unknown key: {key}")
        print(f"Available: {list(known.keys())}")
        return

    print(client.generate_download_instructions(key))


def search_archive(query: str):
    """Search Internet Archive for dictionaries."""
    client = InternetArchiveClient()

    print(f"\nSearching Internet Archive for: {query}")
    print("-" * 50)

    results = client.search_dictionaries(query)

    if results:
        for item in results:
            print(f"\n{item.identifier}")
            print(f"  Title: {item.title}")
            print(f"  Author: {item.creator}")
            print(f"  Year: {item.year}")
            print(f"  URL: {item.url}")
    else:
        print("No results found (or search requires browser)")
        print(f"Try: https://archive.org/search?query={query.replace(' ', '+')}")


def export_johnson(output_path: str):
    """Export curated Johnson entries to JSON."""
    client = JohnsonDictionaryClient()
    client.export_curated_to_json(output_path)


def process_ocr(
    input_path: str,
    output_path: str,
    name: str,
    year: str,
    author: str,
    style: str = "generic"
):
    """Process OCR text file into JSON format."""
    process_ocr_file(
        input_path=input_path,
        output_path=output_path,
        name=name,
        year=int(year),
        author=author,
        style=style
    )


def show_usage():
    """Show usage information."""
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        show_usage()
        return

    command = sys.argv[1]

    if command == "list-sources":
        list_sources()

    elif command == "list-dictionaries":
        list_dictionaries()

    elif command == "fetch-archive":
        if len(sys.argv) < 3:
            print("Usage: fetch-archive <key> [output_dir]")
            print("Run 'list-dictionaries' to see available keys")
            return
        key = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/raw"
        fetch_archive(key, output_dir)

    elif command == "fetch-hathi":
        if len(sys.argv) < 3:
            print("Usage: fetch-hathi <key>")
            print("Run 'list-dictionaries' to see available keys")
            return
        fetch_hathi(sys.argv[2])

    elif command == "search-archive":
        if len(sys.argv) < 3:
            print("Usage: search-archive <query>")
            return
        search_archive(' '.join(sys.argv[2:]))

    elif command == "export-johnson":
        if len(sys.argv) < 3:
            print("Usage: export-johnson <output.json>")
            return
        export_johnson(sys.argv[2])

    elif command == "process-ocr":
        if len(sys.argv) < 7:
            print("Usage: process-ocr <input.txt> <output.json> <name> <year> <author> [style]")
            print("Styles: generic, johnson, bailey")
            return
        style = sys.argv[7] if len(sys.argv) > 7 else "generic"
        process_ocr(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], style)

    else:
        print(f"Unknown command: {command}")
        show_usage()


if __name__ == "__main__":
    main()
