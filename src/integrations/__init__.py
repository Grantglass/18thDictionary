"""Integrations package for fetching dictionary data from external sources."""

from .internet_archive import InternetArchiveClient
from .hathitrust import HathiTrustClient
from .johnson_online import JohnsonDictionaryClient

__all__ = [
    "InternetArchiveClient",
    "HathiTrustClient",
    "JohnsonDictionaryClient",
]
