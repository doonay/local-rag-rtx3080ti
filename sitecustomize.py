"""Automatically isolate Python caches when Python is run from the project root."""

from common.local_storage import configure_local_storage


configure_local_storage()
