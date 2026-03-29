# TODOS

## API / Repos

### P0 — Bug: `repo_path` missing from `/api/repos` response

**Priority:** P0
**Noticed:** gstack /ship on `gstack_1/ship` (2026-03-29)
**Test:** `tests/test_api_repos.py::TestListRepos::test_crest_be_has_repo_path`

**Error:**
```
AssertionError: assert 'repo_path' in {'files': 0, 'functions': 0, 'has_context': True, 'has_summary': True, ...}
```

The `/api/repos` endpoint response for `crest-be` is missing the `repo_path` field. The test expects
the field to exist, be non-empty, and point to an existing filesystem path. The field is likely
not being serialized or populated from `graph.json` in the repos listing handler.

**Fix hint:** Check the repos listing endpoint in `backend/main.py` or the route handler — look for
where repo metadata is assembled and ensure `repo_path` is included from the stored graph data.

---

## Completed

