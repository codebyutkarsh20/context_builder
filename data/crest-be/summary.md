# crest-be — Summary

**Stack:** Alembic, Anthropic, Docker, FastAPI, Gunicorn, OpenAI, Pandas, PostgreSQL, PostgreSQL (async), Pydantic, Python, Redis, SQLAlchemy, Uvicorn, aiohttp, pytest
**Files:** 480 | **LOC:** ~87,503
**Entry Points:** main.py

## Top Hotspots
1. `BaseModel` (class) pagerank=0.0529
2. `get_ist_epoch` (function) pagerank=0.0287
3. `to_dict` (function) pagerank=0.0092
4. `update_from_dict` (function) pagerank=0.0092
5. `update_timestamp` (function) pagerank=0.0092
6. `create_new` (function) pagerank=0.0092
7. `__repr__` (function) pagerank=0.0092
8. `app/db/base_class.py` (file) pagerank=0.0063
9. `post_call` (function) pagerank=0.0059
10. `app/utils/logging_util.py` (file) pagerank=0.0054

## Business Rules
- [endpoint] API Endpoint: PATCH /target/{target_id} → handler: update_user_category_target()
- [endpoint] API Endpoint: GET /summary → handler: get_holdings_summary()
- [endpoint] API Endpoint: GET /fund-matrix → handler: get_overlap_matrix()
- [endpoint] API Endpoint: GET /securities → handler: get_securities_overlap()
- [endpoint] API Endpoint: GET /family → handler: get_family_overlap_analysis()
- [endpoint] API Endpoint: GET /key-metrics → handler: get_key_metrics()
- [endpoint] API Endpoint: GET /dividends → handler: get_dividends()
- [endpoint] API Endpoint: GET /benchmark → handler: get_benchmark_comparison()
- [endpoint] API Endpoint: GET /metrics → handler: get_risk_metrics()
- [endpoint] API Endpoint: GET /metrics/chart → handler: get_risk_metrics_chart()
- [endpoint] API Endpoint: GET /contributors → handler: get_top_risk_contributors()
- [endpoint] API Endpoint: GET /{asset_class_id}/form-fields → handler: get_form_fields_by_asset_class_id()
- [endpoint] API Endpoint: POST / → handler: create_asset_group()
- [endpoint] API Endpoint: GET / → handler: get_asset_groups()
- [endpoint] API Endpoint: GET /{asset_group_id} → handler: get_asset_group()
- [endpoint] API Endpoint: PUT /{asset_group_id} → handler: update_asset_group()
- [endpoint] API Endpoint: DELETE /{asset_group_id} → handler: delete_asset_group()
- [endpoint] API Endpoint: POST /signup → handler: signup()
- [endpoint] API Endpoint: POST /generate-otp → handler: generate_otp()
- [endpoint] API Endpoint: POST /verify-otp → handler: verify_otp()