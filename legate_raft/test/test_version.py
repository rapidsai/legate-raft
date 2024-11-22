import legate_raft


def test_version_constants_are_populated():
    # __version__ should always be non-empty
    assert isinstance(legate_raft.__version__, str)
    assert len(legate_raft.__version__) > 0
