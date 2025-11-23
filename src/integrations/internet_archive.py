"""
Internet Archive integration for downloading historical dictionary texts.

The Internet Archive (archive.org) hosts scanned copies of many 18th century
dictionaries. This module provides tools to:
1. Search for dictionary items
2. Download available text formats (OCR text, DjVu, PDF)
3. Extract and process the text content
"""

import json
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import urllib.request
import urllib.parse
import urllib.error


@dataclass
class ArchiveItem:
    """Represents an item from Internet Archive."""
    identifier: str
    title: str
    creator: str
    year: int | None
    description: str
    formats: list[str]

    @property
    def url(self) -> str:
        return f"https://archive.org/details/{self.identifier}"

    @property
    def metadata_url(self) -> str:
        return f"https://archive.org/metadata/{self.identifier}"


class InternetArchiveClient:
    """
    Client for Internet Archive API.

    Usage:
        client = InternetArchiveClient()

        # Search for dictionaries
        items = client.search_dictionaries("Johnson dictionary 1755")

        # Get item details
        item = client.get_item("johnsons_dictionary_1755")

        # Download text
        text = client.download_text(item)
    """

    BASE_URL = "https://archive.org"
    SEARCH_URL = "https://archive.org/advancedsearch.php"

    # Known 18th century dictionary identifiers
    KNOWN_DICTIONARIES = {
        "johnson_1755_v1": "dictionaryofengl01johnuoft",
        "johnson_1755_v2": "dictionaryofengl02johnuoft",
        "johnson_1773": "johnsons_dictionary_1755",
        "johnson_1785": "dictionaryofengl1785john",
        "bailey_1721": "universaletymolo00bail",
        "bailey_1737": "universaletymolo00bailuoft",
        "walker_1791": "criticalpronounc00walkuoft",
        "sheridan_1780": "generaldictontic00shergoog",
    }

    def __init__(self, cache_dir: str | Path = "data/cache/archive"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = 1.0  # seconds between requests
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _fetch_json(self, url: str) -> dict:
        """Fetch JSON from URL with rate limiting."""
        self._rate_limit()

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "18thCenturyDictionaryProject/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason}")
            raise
        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason}")
            raise

    def _fetch_text(self, url: str) -> str:
        """Fetch text content from URL."""
        self._rate_limit()

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "18thCenturyDictionaryProject/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read().decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            raise

    def search_dictionaries(
        self,
        query: str,
        max_results: int = 20
    ) -> list[ArchiveItem]:
        """
        Search Internet Archive for dictionary items.

        Args:
            query: Search query (e.g., "Johnson dictionary 1755")
            max_results: Maximum number of results to return

        Returns:
            List of ArchiveItem objects
        """
        # Build search query
        full_query = f'({query}) AND mediatype:texts'

        params = {
            'q': full_query,
            'fl[]': ['identifier', 'title', 'creator', 'date', 'description'],
            'sort[]': 'downloads desc',
            'rows': str(max_results),
            'page': '1',
            'output': 'json'
        }

        query_string = urllib.parse.urlencode(params, doseq=True)
        url = f"{self.SEARCH_URL}?{query_string}"

        data = self._fetch_json(url)

        items = []
        for doc in data.get('response', {}).get('docs', []):
            # Parse year from date
            year = None
            date_str = doc.get('date', '')
            if date_str:
                year_match = re.search(r'\b(1[67]\d{2})\b', date_str)
                if year_match:
                    year = int(year_match.group(1))

            item = ArchiveItem(
                identifier=doc.get('identifier', ''),
                title=doc.get('title', 'Unknown'),
                creator=doc.get('creator', 'Unknown'),
                year=year,
                description=doc.get('description', '')[:500] if doc.get('description') else '',
                formats=[]  # Will be filled by get_item
            )
            items.append(item)

        return items

    def get_item(self, identifier: str) -> ArchiveItem | None:
        """
        Get detailed information about an archive item.

        Args:
            identifier: Archive.org item identifier

        Returns:
            ArchiveItem with full details including available formats
        """
        url = f"{self.BASE_URL}/metadata/{identifier}"

        try:
            data = self._fetch_json(url)
        except Exception:
            return None

        metadata = data.get('metadata', {})
        files = data.get('files', [])

        # Get available formats
        formats = list(set(f.get('format', '') for f in files if f.get('format')))

        # Parse year
        year = None
        date_str = metadata.get('date', '')
        if date_str:
            year_match = re.search(r'\b(1[67]\d{2})\b', str(date_str))
            if year_match:
                year = int(year_match.group(1))

        return ArchiveItem(
            identifier=identifier,
            title=metadata.get('title', 'Unknown'),
            creator=metadata.get('creator', 'Unknown'),
            year=year,
            description=metadata.get('description', '')[:500] if metadata.get('description') else '',
            formats=formats
        )

    def get_text_file_url(self, identifier: str) -> str | None:
        """
        Get URL for the best available text file.

        Prefers: Full text > DjVu TXT > OCR text
        """
        url = f"{self.BASE_URL}/metadata/{identifier}"

        try:
            data = self._fetch_json(url)
        except Exception:
            return None

        files = data.get('files', [])

        # Priority order for text files
        text_extensions = ['_djvu.txt', '.txt', '_text.pdf']

        for ext in text_extensions:
            for f in files:
                name = f.get('name', '')
                if name.endswith(ext):
                    return f"{self.BASE_URL}/download/{identifier}/{name}"

        return None

    def download_text(
        self,
        identifier: str,
        use_cache: bool = True
    ) -> str | None:
        """
        Download the text content of a dictionary.

        Args:
            identifier: Archive.org item identifier
            use_cache: Whether to use cached version if available

        Returns:
            Text content or None if not available
        """
        cache_path = self.cache_dir / f"{identifier}.txt"

        # Check cache
        if use_cache and cache_path.exists():
            print(f"Using cached text for {identifier}")
            return cache_path.read_text(encoding='utf-8')

        # Get text file URL
        text_url = self.get_text_file_url(identifier)
        if not text_url:
            print(f"No text file found for {identifier}")
            return None

        print(f"Downloading text from {text_url}...")

        try:
            text = self._fetch_text(text_url)

            # Cache the result
            cache_path.write_text(text, encoding='utf-8')

            return text
        except Exception as e:
            print(f"Error downloading text: {e}")
            return None

    def list_known_dictionaries(self) -> dict[str, str]:
        """Return dictionary of known 18th century dictionary identifiers."""
        return self.KNOWN_DICTIONARIES.copy()

    def download_known_dictionary(
        self,
        key: str,
        use_cache: bool = True
    ) -> str | None:
        """
        Download a known dictionary by its key.

        Args:
            key: Key from KNOWN_DICTIONARIES (e.g., "johnson_1755_v1")
            use_cache: Whether to use cached version

        Returns:
            Text content or None
        """
        if key not in self.KNOWN_DICTIONARIES:
            print(f"Unknown dictionary key: {key}")
            print(f"Available: {list(self.KNOWN_DICTIONARIES.keys())}")
            return None

        identifier = self.KNOWN_DICTIONARIES[key]
        return self.download_text(identifier, use_cache=use_cache)
