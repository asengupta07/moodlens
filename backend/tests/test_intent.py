"""Smoke tests for the regex-fallback path of intent_parser, with the new
permanent unlearning fields."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent_parser import _regex_parse


def _toy_db():
    return pd.DataFrame({
        "title": ["Inception", "The Conjuring", "Old Movie"],
        "title_clean": ["inception", "the conjuring", "old movie"],
        "genres_list": [["Action", "Sci-Fi"], ["Horror"], ["Drama"]],
        "cast_list": [[], [], []],
        "directors_list": [[], [], []],
    })


def test_session_dislike_not_permanent():
    r = _regex_parse("I hate horror", _toy_db())
    assert r["is_permanent"] is False
    # disliked_genres may include Horror from session-style hate
    assert r["permanent_genre_block"] == []


def test_permanent_genre_block():
    r = _regex_parse("Block horror forever, never show me horror again", _toy_db())
    assert r["is_permanent"] is True
    assert "Horror" in r["permanent_genre_block"]


def test_permanent_year_block():
    r = _regex_parse("never show me movies before 1990 ever again", _toy_db())
    assert r["is_permanent"] is True
    assert r["permanent_year_block"] == 1990


def test_pure_request():
    r = _regex_parse("recommend me action movies", _toy_db())
    # "recommend" matches the positive sentiment pattern → Action lands in liked_genres
    assert ("Action" in r["genres_requested"]) or ("Action" in r["liked_genres"])
    assert r["is_permanent"] is False

    r2 = _regex_parse("show me thrillers", _toy_db())
    assert ("Thriller" in r2["genres_requested"]) or ("Thriller" in r2["liked_genres"])


if __name__ == "__main__":
    test_session_dislike_not_permanent()
    print("✓ session dislike not permanent")
    test_permanent_genre_block()
    print("✓ permanent genre block detected")
    test_permanent_year_block()
    print("✓ permanent year block detected")
    test_pure_request()
    print("✓ pure request parses cleanly")
    print("\nAll intent regex tests passed.")
