# context_builder — Summary

**Stack:** Anthropic, Docker, FastAPI, LangChain, Node.js, Pydantic, Python, React, Tailwind CSS, TypeScript, Uvicorn, Vite
**Files:** 133 | **LOC:** ~22,182
**Entry Points:** backend/main.py

## Top Hotspots
1. `_scan_body` (function) pagerank=0.0198
2. `classify_condition` (function) pagerank=0.0196
3. `analyze_query` (function) pagerank=0.0170
4. `_should_skip_dir` (function) pagerank=0.0153
5. `check_file` (function) pagerank=0.0139
6. `_flags_path` (function) pagerank=0.0129
7. `_redact_secrets` (function) pagerank=0.0127
8. `_load_repo_rules` (function) pagerank=0.0124
9. `_load_flags` (function) pagerank=0.0101
10. `_make_extractor` (function) pagerank=0.0094

## Business Rules
- [docstring] The reviewer must NOT see the developer's source_code to prevent inherited bias.
- [constant] RECENT_COMMITS_LIMIT = 50
- [constant] HOTSPOT_LIMIT = 10
- [constant] GIT_TIMEOUT = 30
- [constant] BATCH_SIZE = 5
- [docstring] Must be called while holding _agent_jobs_lock.
- [endpoint] API Endpoint: POST /agent/run → handler: run_agent()
- [endpoint] API Endpoint: GET /agent/status/{job_id} → handler: get_agent_status()
- [endpoint] API Endpoint: GET /agent/jobs → handler: list_agent_jobs()
- [endpoint] API Endpoint: GET /agent/tickets → handler: list_mock_tickets()
- [endpoint] API Endpoint: POST /agent/run-mock/{ticket_id} → handler: run_mock_ticket()
- [endpoint] API Endpoint: POST /chat → handler: chat()
- [endpoint] API Endpoint: GET /context/layers → handler: get_context_layers()
- [endpoint] API Endpoint: GET /context/summary → handler: get_context_summary()
- [endpoint] API Endpoint: GET /context/full → handler: get_context_full()
- [endpoint] API Endpoint: POST /eval/{repo}/run → handler: run_eval_endpoint()
- [endpoint] API Endpoint: GET /eval/{repo}/results → handler: get_eval_results()
- [endpoint] API Endpoint: GET /flags/{repo} → handler: api_list_flags()
- [endpoint] API Endpoint: POST /flags/{repo}/{flag_name}/toggle → handler: api_toggle_flag()
- [endpoint] API Endpoint: GET /flags/{repo}/{flag_name} → handler: api_get_flag()