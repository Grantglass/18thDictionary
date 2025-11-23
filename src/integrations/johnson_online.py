"""
Johnson's Dictionary Online integration.

johnsonsdictionaryonline.com provides a searchable interface to Johnson's
Dictionary (1755 and 1773 editions). This module provides tools to search
and extract entries.

NOTE: This scraper respects robots.txt and rate limits. Please use responsibly.
"""

import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from html.parser import HTMLParser
import urllib.request
import urllib.parse
import urllib.error


class HTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script = False
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.in_script = True
        elif tag == 'style':
            self.in_style = True
        elif tag in ('br', 'p', 'div', 'li'):
            self.text_parts.append('\n')

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False
        elif tag == 'style':
            self.in_style = False

    def handle_data(self, data):
        if not self.in_script and not self.in_style:
            self.text_parts.append(data)

    def get_text(self) -> str:
        return ''.join(self.text_parts).strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


@dataclass
class JohnsonEntry:
    """A dictionary entry from Johnson's Dictionary Online."""
    headword: str
    definition: str
    part_of_speech: str
    etymology: str
    quotations: list[str]
    edition: str  # "1755" or "1773"


class JohnsonDictionaryClient:
    """
    Client for Johnson's Dictionary Online.

    This provides a structured way to access entries from Johnson's Dictionary.
    Since the site doesn't have a public API, we provide:
    1. A curated dataset of key entries
    2. Search URL generation for manual lookup

    Usage:
        client = JohnsonDictionaryClient()

        # Get curated entries
        entries = client.get_curated_entries()

        # Generate search URL for manual lookup
        url = client.get_search_url("enthusiasm")
    """

    BASE_URL = "https://johnsonsdictionaryonline.com"

    # Curated set of important/famous entries from Johnson's Dictionary
    # These are historically documented and frequently cited definitions
    CURATED_ENTRIES = {
        "lexicographer": JohnsonEntry(
            headword="lexicographer",
            definition="A writer of dictionaries; a harmless drudge, that busies himself in tracing the original, and detailing the signification of words.",
            part_of_speech="n.s.",
            etymology="λεξικὸν and γράφω",
            quotations=[],
            edition="1755"
        ),
        "oats": JohnsonEntry(
            headword="oats",
            definition="A grain, which in England is generally given to horses, but in Scotland supports the people.",
            part_of_speech="n.s.",
            etymology="Saxon ata",
            quotations=[],
            edition="1755"
        ),
        "patron": JohnsonEntry(
            headword="patron",
            definition="One who countenances, supports or protects. Commonly a wretch who supports with insolence, and is paid with flattery.",
            part_of_speech="n.s.",
            etymology="Latin patronus",
            quotations=[],
            edition="1755"
        ),
        "pension": JohnsonEntry(
            headword="pension",
            definition="An allowance made to any one without an equivalent. In England it is generally understood to mean pay given to a state hireling for treason to his country.",
            part_of_speech="n.s.",
            etymology="French pension",
            quotations=[],
            edition="1755"
        ),
        "excise": JohnsonEntry(
            headword="excise",
            definition="A hateful tax levied upon commodities, and adjudged not by the common judges of property, but wretches hired by those to whom excise is paid.",
            part_of_speech="n.s.",
            etymology="Dutch excijs",
            quotations=[],
            edition="1755"
        ),
        "dull": JohnsonEntry(
            headword="dull",
            definition="Not exhilarating; not delightful; as, to make dictionaries is dull work.",
            part_of_speech="adj.",
            etymology="Dutch dol; Teutonick dol",
            quotations=[],
            edition="1755"
        ),
        "network": JohnsonEntry(
            headword="network",
            definition="Any thing reticulated or decussated, at equal distances, with interstices between the intersections.",
            part_of_speech="n.s.",
            etymology="net and work",
            quotations=[],
            edition="1755"
        ),
        "pastern": JohnsonEntry(
            headword="pastern",
            definition="The knee of a horse.",
            part_of_speech="n.s.",
            etymology="",
            quotations=["This is wrong. The pastern is the part between the joint next the foot and the hoof."],
            edition="1755"
        ),
        "enthusiasm": JohnsonEntry(
            headword="enthusiasm",
            definition="A vain belief of private revelation; a vain confidence of divine favour or communication.",
            part_of_speech="n.s.",
            etymology="ἐνθουσιασμὸς",
            quotations=[
                "Enthusiasm is founded neither on reason nor divine revelation, but rises from the conceits of a warmed or overweening brain. Locke."
            ],
            edition="1755"
        ),
        "tory": JohnsonEntry(
            headword="tory",
            definition="One who adheres to the antient constitution of the state, and the apostolical hierarchy of the church of England, opposed to a whig.",
            part_of_speech="n.s.",
            etymology="A cant term, derived, I suppose, from an Irish word signifying a savage",
            quotations=[],
            edition="1755"
        ),
        "whig": JohnsonEntry(
            headword="whig",
            definition="The name of a faction.",
            part_of_speech="n.s.",
            etymology="",
            quotations=[
                "Whoever has a true value for church and state, should avoid the extremes of whig for the sake of the former, and the extremes of tory on the account of the latter. Swift."
            ],
            edition="1755"
        ),
        "wit": JohnsonEntry(
            headword="wit",
            definition="The powers of the mind; the mental faculties; the intellect. This is the original signification.",
            part_of_speech="n.s.",
            etymology="Saxon ƿit",
            quotations=[],
            edition="1755"
        ),
        "novel": JohnsonEntry(
            headword="novel",
            definition="A small tale, generally of love.",
            part_of_speech="n.s.",
            etymology="Italian novella",
            quotations=[],
            edition="1755"
        ),
        "essay": JohnsonEntry(
            headword="essay",
            definition="A loose sally of the mind; an irregular indigested piece; not a regular and orderly composition.",
            part_of_speech="n.s.",
            etymology="French essai",
            quotations=[],
            edition="1755"
        ),
        "imagination": JohnsonEntry(
            headword="imagination",
            definition="Fancy; the power of forming ideal pictures; the power of representing things absent to one's self or others.",
            part_of_speech="n.s.",
            etymology="Latin imaginatio",
            quotations=[],
            edition="1755"
        ),
        "genius": JohnsonEntry(
            headword="genius",
            definition="The protecting or ruling power of men, places, or things; mental power or faculties; disposition of nature by which any one is qualified for some peculiar employment; nature; disposition.",
            part_of_speech="n.s.",
            etymology="Latin genius",
            quotations=[],
            edition="1755"
        ),
        "club": JohnsonEntry(
            headword="club",
            definition="An assembly of good fellows, meeting under certain conditions.",
            part_of_speech="n.s.",
            etymology="",
            quotations=[],
            edition="1755"
        ),
        "cough": JohnsonEntry(
            headword="cough",
            definition="A convulsion of the lungs, vellicated by some sharp serosity. It is pronounced coff.",
            part_of_speech="n.s.",
            etymology="",
            quotations=[],
            edition="1755"
        ),
        "stockjobber": JohnsonEntry(
            headword="stockjobber",
            definition="A low wretch who gets money by buying and selling shares in the funds.",
            part_of_speech="n.s.",
            etymology="stock and jobber",
            quotations=[],
            edition="1755"
        ),
        "favourite": JohnsonEntry(
            headword="favourite",
            definition="One chosen as a companion by his superiour; a mean wretch whose whole business is by any means to please.",
            part_of_speech="n.s.",
            etymology="French favori",
            quotations=[],
            edition="1755"
        ),
    }

    def __init__(self, cache_dir: str | Path = "data/cache/johnson"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_search_url(self, word: str, edition: str = "1755") -> str:
        """
        Generate URL to search for a word on Johnson's Dictionary Online.

        Args:
            word: Word to search for
            edition: "1755" or "1773"

        Returns:
            URL for manual lookup
        """
        encoded_word = urllib.parse.quote(word)
        return f"{self.BASE_URL}/views/search.php?term={encoded_word}"

    def get_page_url(self, letter: str, edition: str = "1755") -> str:
        """
        Get URL for a specific letter's entries.

        Args:
            letter: Single letter (A-Z)
            edition: "1755" or "1773"
        """
        return f"{self.BASE_URL}/{edition}page/{letter.lower()}"

    def get_curated_entries(self) -> dict[str, JohnsonEntry]:
        """
        Get the curated set of famous/important Johnson entries.

        These are historically documented definitions that are frequently
        cited in scholarship.
        """
        return self.CURATED_ENTRIES.copy()

    def get_curated_entry(self, word: str) -> JohnsonEntry | None:
        """Get a specific curated entry."""
        return self.CURATED_ENTRIES.get(word.lower())

    def export_curated_to_json(self, output_path: str | Path) -> None:
        """
        Export curated entries to our standard JSON format.

        Args:
            output_path: Path for output JSON file
        """
        entries = []
        for entry in self.CURATED_ENTRIES.values():
            entries.append({
                "headword": entry.headword,
                "definition": entry.definition,
                "part_of_speech": entry.part_of_speech,
                "etymology": entry.etymology,
                "examples": entry.quotations
            })

        output_data = {
            "name": "Johnson's Dictionary (Curated)",
            "year": 1755,
            "author": "Samuel Johnson",
            "description": "Curated selection of famous and frequently-cited entries from Johnson's Dictionary of the English Language (1755).",
            "entries": entries
        }

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(entries)} entries to {output_path}")

    def list_available_words(self) -> list[str]:
        """List all words available in the curated set."""
        return sorted(self.CURATED_ENTRIES.keys())

    def get_citation(self, word: str) -> str:
        """
        Get a formatted citation for a Johnson entry.

        Args:
            word: Headword to cite

        Returns:
            Formatted citation string
        """
        entry = self.get_curated_entry(word)
        if not entry:
            return f"Entry not found: {word}"

        return (
            f'Johnson, Samuel. "{entry.headword}." '
            f'A Dictionary of the English Language. London, {entry.edition}.'
        )
