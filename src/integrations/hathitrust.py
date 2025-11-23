"""
HathiTrust Digital Library integration.

HathiTrust provides access to millions of digitized volumes from research
libraries. This module interfaces with their APIs to search for and
access historical dictionary texts.

API Documentation: https://www.hathitrust.org/data_api
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import urllib.request
import urllib.parse
import urllib.error


@dataclass
class HathiVolume:
    """Represents a volume from HathiTrust."""
    htid: str  # HathiTrust ID (e.g., "mdp.39015028036104")
    title: str
    author: str
    publisher: str
    year: int | None
    rights: str  # "pd" (public domain), "ic" (in copyright), etc.
    url: str

    @property
    def is_public_domain(self) -> bool:
        return self.rights in ("pd", "pdus", "pd-google")

    @property
    def page_reader_url(self) -> str:
        """URL to read the volume page by page."""
        return f"https://babel.hathitrust.org/cgi/pt?id={self.htid}"


class HathiTrustClient:
    """
    Client for HathiTrust APIs.

    Provides access to:
    1. Bibliographic API - Search and metadata
    2. Data API - Page images and OCR text (for public domain works)

    Usage:
        client = HathiTrustClient()

        # Search for dictionaries
        volumes = client.search("Johnson dictionary English")

        # Get volume details
        volume = client.get_volume("mdp.39015028036104")

        # Download OCR text (public domain only)
        pages = client.get_page_text(volume, page_range=(1, 50))
    """

    SOLR_URL = "https://catalog.hathitrust.org/api/volumes"
    DATA_API_URL = "https://babel.hathitrust.org/cgi/htd"

    # Known 18th century dictionary HathiTrust IDs
    KNOWN_DICTIONARIES = {
        "johnson_1755_v1": "ucm.5326809190",  # Volume 1 A-K
        "johnson_1755_v2": "ucm.5326809191",  # Volume 2 L-Z
        "johnson_1785": "mdp.39015028036104",
        "bailey_1721": "mdp.39015002188498",
        "bailey_1730": "mdp.39015028036112",
        "walker_1791": "nyp.33433082379551",
        "sheridan_1780": "mdp.39015028036096",
        "dyche_1740": "mdp.39015028036088",  # Dyche & Pardon's New General English Dictionary
    }

    def __init__(self, cache_dir: str | Path = "data/cache/hathitrust"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = 1.0
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _fetch_json(self, url: str) -> dict:
        """Fetch JSON from URL."""
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

    def search(
        self,
        query: str,
        max_results: int = 20
    ) -> list[HathiVolume]:
        """
        Search HathiTrust catalog.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of HathiVolume objects
        """
        # Use the brief API for searching
        params = {
            'q': query,
            'searchtype': 'all',
            'setop': 'all',
        }

        # HathiTrust uses a different search interface
        # We'll use the OCLC lookup as a workaround for simple searches
        search_url = f"https://catalog.hathitrust.org/Search/Home?lookfor={urllib.parse.quote(query)}&type=all&submit=Find"

        # For now, return empty and guide users to use known dictionaries
        print(f"Search URL (open in browser): {search_url}")
        print("Note: HathiTrust search requires browser interaction.")
        print("Use list_known_dictionaries() for pre-identified dictionary volumes.")

        return []

    def get_volume_by_htid(self, htid: str) -> HathiVolume | None:
        """
        Get volume information by HathiTrust ID.

        Args:
            htid: HathiTrust volume ID (e.g., "mdp.39015028036104")

        Returns:
            HathiVolume object or None
        """
        url = f"{self.SOLR_URL}/brief/htid/{htid}.json"

        try:
            data = self._fetch_json(url)
        except Exception as e:
            print(f"Error fetching volume {htid}: {e}")
            return None

        if 'items' not in data or not data['items']:
            return None

        item = data['items'][0]
        record = data.get('records', {}).get(item.get('fromRecord', ''), {})

        # Parse year
        year = None
        pub_dates = record.get('publishDates', [])
        if pub_dates:
            try:
                year = int(pub_dates[0])
            except (ValueError, IndexError):
                pass

        return HathiVolume(
            htid=htid,
            title=record.get('titles', ['Unknown'])[0] if record.get('titles') else 'Unknown',
            author=record.get('authors', ['Unknown'])[0] if record.get('authors') else 'Unknown',
            publisher=record.get('publisher', ['Unknown'])[0] if record.get('publisher') else 'Unknown',
            year=year,
            rights=item.get('rightsCode', 'unknown'),
            url=f"https://babel.hathitrust.org/cgi/pt?id={htid}"
        )

    def list_known_dictionaries(self) -> dict[str, str]:
        """Return dictionary of known 18th century dictionary HathiTrust IDs."""
        return self.KNOWN_DICTIONARIES.copy()

    def get_known_dictionary(self, key: str) -> HathiVolume | None:
        """
        Get a known dictionary volume.

        Args:
            key: Key from KNOWN_DICTIONARIES

        Returns:
            HathiVolume or None
        """
        if key not in self.KNOWN_DICTIONARIES:
            print(f"Unknown dictionary key: {key}")
            print(f"Available: {list(self.KNOWN_DICTIONARIES.keys())}")
            return None

        htid = self.KNOWN_DICTIONARIES[key]
        return self.get_volume_by_htid(htid)

    def get_page_image_url(
        self,
        htid: str,
        page: int,
        size: str = "full"
    ) -> str:
        """
        Get URL for a page image.

        Args:
            htid: HathiTrust volume ID
            page: Page number (1-indexed)
            size: "thumbnail", "full", or pixel width

        Returns:
            URL for the page image
        """
        return f"https://babel.hathitrust.org/cgi/imgsrv/image?id={htid};seq={page};size={size}"

    def get_volume_structure(self, htid: str) -> dict | None:
        """
        Get the structure (page count, etc.) of a volume.

        Note: Requires the Data API which needs authentication for full access.
        """
        url = f"{self.DATA_API_URL}/structure/{htid}"

        try:
            data = self._fetch_json(url)
            return data
        except Exception as e:
            print(f"Error getting structure for {htid}: {e}")
            print("Note: Full text access may require HathiTrust Data API credentials.")
            return None

    def generate_download_instructions(self, key: str) -> str:
        """
        Generate instructions for manually downloading a dictionary.

        Args:
            key: Key from KNOWN_DICTIONARIES

        Returns:
            Instructions string
        """
        if key not in self.KNOWN_DICTIONARIES:
            return f"Unknown dictionary key: {key}"

        htid = self.KNOWN_DICTIONARIES[key]
        volume = self.get_volume_by_htid(htid)

        if not volume:
            return f"Could not fetch volume information for {key}"

        instructions = f"""
Download Instructions for: {volume.title}
{'=' * 60}

HathiTrust ID: {htid}
Author: {volume.author}
Year: {volume.year}
Rights: {volume.rights}

1. Page Reader (browse online):
   {volume.page_reader_url}

2. Download Options:
"""
        if volume.is_public_domain:
            instructions += f"""
   - PDF Download: https://babel.hathitrust.org/cgi/pt?id={htid}&view=pdf
   - Plain Text: https://babel.hathitrust.org/cgi/pt?id={htid}&view=plaintext

   For bulk download, use HathiTrust's Data API:
   https://www.hathitrust.org/data_api

3. After downloading, use our processing scripts:
   python scripts/process_data.py process-ocr <input.txt> <output.json> "{volume.title}" {volume.year or 1750} "{volume.author}"
"""
        else:
            instructions += """
   This volume is NOT in the public domain.
   - Online viewing only (page by page)
   - Full text download not available

   Consider using Internet Archive for alternative copies.
"""

        return instructions

    def export_all_info(self, output_path: str | Path) -> None:
        """
        Export information about all known dictionaries to a JSON file.
        """
        info = {}

        for key, htid in self.KNOWN_DICTIONARIES.items():
            print(f"Fetching info for {key}...")
            volume = self.get_volume_by_htid(htid)

            if volume:
                info[key] = {
                    "htid": volume.htid,
                    "title": volume.title,
                    "author": volume.author,
                    "year": volume.year,
                    "rights": volume.rights,
                    "is_public_domain": volume.is_public_domain,
                    "url": volume.url,
                    "page_reader_url": volume.page_reader_url
                }
            else:
                info[key] = {"htid": htid, "error": "Could not fetch"}

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        print(f"Exported info to {output_path}")
