"""Shared ingestion helpers: fetching, hashing, seat-key normalization."""
from __future__ import annotations

import hashlib

import httpx

STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
}


def fetch(url: str, timeout: float = 120.0, headers: dict | None = None) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.content


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def house_seat_key(state: str, district: int | str) -> str:
    """house-{ST}-{NN}; at-large districts (0 or 1) normalize to 01."""
    number = int(district) if str(district).strip() else 1
    if number == 0:
        number = 1
    return f"house-{state}-{number:02d}"


def senate_seat_key(state: str, special: bool = False) -> str:
    return f"senate-{state}" + ("-special" if special else "")


def two_party_margin(dem_votes: float, rep_votes: float) -> float | None:
    total = dem_votes + rep_votes
    if total <= 0:
        return None
    return (dem_votes - rep_votes) / total * 100.0
