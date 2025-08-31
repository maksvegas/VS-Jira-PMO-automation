#!/usr/bin/env python3
# jira_to_teams_report.py — grouped Jira → Teams report
# Version v2.5 (with force-get option, parent priority rank support)

import os, sys, json
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---------------- Helpers ----------------

def env_or_config(name: str, default=None):
    return os.getenv(name, default)

def parse_bool(v: Optional[str], default=False) -> bool:
    if v is None: return default
    return v.strip().lower() in {"1","true","yes","y","on"}

def parse_reconcile_ids(csv: str) -> Optional[List[int]]:
    if not csv: return None
    ids = []
    for token in csv.split(","):
        t = token.strip()
        if t and t.isdigit():
            ids.append(int(t))
    return ids or None

def jira_auth(email: str, token: str):
    return {"Accept":"application/json","Content-Type":"application/json"}, (email, token)

def detect_fields(base: str, email: str, token: str, debug=False) -> Dict[str, str]:
    found = {"rank": "", "epic_rank": ""}
    try:
        url = f"{base}/rest/api/3/field"
        headers, auth = jira_auth(email, token)
        r = requests.get(url, headers=headers, auth=auth, timeout=60)
        if r.status_code != 200:
            if debug: print(f"Field discovery failed: {r.status_code}")
            return found
        arr = r.json()
        for f in arr:
            name = (f.get("name") or "").strip().lower()
            fid = f.get("id") or ""
            schema = f.get("schema") or {}
            c = (schema.get("custom") or "").lower()
            if "lexorank" in c or "rank" in name or "rank" in fid.lower():
                found["rank"] = fid
            if "epic rank" in name or "epicrank" in c:
                found["epic_rank"] = fid
        if debug:
            print(f"Field discovery -> rank={found['rank'] or '(none)'} epic_rank={found['epic_rank'] or '(none)'}")
    except Exception as e:
        if debug: print(f"Field discovery exception: {e}")
    return found

def compute_request_fields(fields: List[str], group_by_parent: bool, issue_rank_field: str, child_priority_field: str) -> List[str]:
    req = [f for f in fields if f.lower() != "key"]
    if any(f.lower() in {"parentsummary","parent_summary"} for f in fields) or group_by_parent:
        if "parent" not in {x.lower() for x in req}:
            req.append("parent")
    if issue_rank_field and issue_rank_field not in req:
        req.append(issue_rank_field)
    if child_priority_field and child_priority_field not in req:
        req.append(child_priority_field)
    return req

def format_date(dt_str: str, tzname: str, fmt: str) -> str:
    if not dt_str: return ""
    try:
        s = dt_str.replace("+0000","+00:00")
        if s.endswith("Z"): s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if tzname and ZoneInfo: dt = dt.astimezone(ZoneInfo(tzname))
        return dt.strftime(fmt)
    except Exception:
        return dt_str

# ------------- Fetch ------------

def fetch_enhanced(base: str, email: str, token: str, jql: str, req_fields: List[str],
                   max_issues: int, batch_size: int, expand: str,
                   reconcile_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
    url = f"{base}/rest/api/3/search/jql"
    headers, auth = jira_auth(email, token)
    out: List[Dict[str, Any]] = []
    next_token: Optional[str] = None
    while True:
        remaining = max_issues - len(out)
        if remaining <= 0: break
        limit = min(batch_size, remaining)
        payload: Dict[str, Any] = {"jql": jql, "maxResults": limit, "fields": req_fields}
        if expand: payload["expand"] = expand
        if reconcile_ids: payload["reconcileIssues"] = reconcile_ids
        if next_token: payload["nextPageToken"] = next_token
        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        batch = data.get("issues", [])
        out.extend(batch)
        next_token = data.get("nextPageToken")
        if not next_token or not batch: break
    return out

def fetch_legacy(base: str, email: str, token: str, jql: str, req_fields: List[str],
                 max_issues: int, batch_size: int, expand: str) -> List[Dict[str, Any]]:
    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    out: List[Dict[str, Any]] = []
    start_at = 0
    while True:
        remaining = max_issues - len(out)
        if remaining <= 0: break
        limit = min(batch_size, remaining)
        payload: Dict[str, Any] = {"jql": jql, "startAt": start_at, "maxResults": limit, "fields": req_fields}
        if expand: payload["expand"] = [e.strip() for e in expand.split(",") if e.strip()]
        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        batch = data.get("issues", [])
        out.extend(batch)
        if not batch: break
        start_at += len(batch)
    return out

# ------------- Parent enrichment ------------

def fetch_parent_fields_individually(base: str, email: str, token: str, keys: list, fields: list, debug: bool=False) -> dict:
    headers, auth = jira_auth(email, token)
    out = {}
    for k in sorted(keys):
        try:
            url = f"{base}/rest/api/3/issue/{k}?fields={','.join(fields)}"
            r = requests.get(url, headers=headers, auth=auth, timeout=45)
            if r.status_code == 200:
                flds = (r.json().get("fields") or {})
                out[k] = {fld: flds.get(fld) for fld in fields}
            else:
                if debug: print(f"Fallback GET /issue/{k} failed: {r.status_code}")
        except Exception as e:
            if debug: print(f"Fallback GET /issue/{k} exception: {e}")
    return out

def bulk_fill_parent_summaries(base: str, email: str, token: str, issues: List[Dict[str, Any]],
                               parent_rank_field: str, epic_rank_field: str,
                               parent_priority_field: str = "",
                               debug: bool=False, force_get: bool=False) -> None:
    want_keys = set()
    for iss in issues:
        fields = iss.get("fields") or {}
        parent = fields.get("parent")
        if isinstance(parent, dict) and parent.get("key"):
            want_keys.add(parent["key"])
    if not want_keys: return

    headers, auth = jira_auth(email, token)
    if force_get:
        if debug:
            print(f"Parent enrichment: FORCE GET for {len(want_keys)} parents")
        want_fields = ["summary"]
        if parent_priority_field: want_fields.append(parent_priority_field)
        if parent_rank_field:     want_fields.append(parent_rank_field)
        if epic_rank_field:       want_fields.append(epic_rank_field)
        per = fetch_parent_fields_individually(base, email, token, want_keys, want_fields, debug=debug)
        for iss in issues:
            p = (iss.get("fields") or {}).get("parent")
            if isinstance(p, dict) and p.get("key"):
                pf = per.get(p["key"], {})
                p["_summary_cache"] = pf.get("summary", "")
                if parent_priority_field: p["_priority_num_cache"] = pf.get(parent_priority_field)
        return

    # Bulk path
    url = f"{base}/rest/api/3/search/jql"
    fetch_fields = ["summary"]
    if parent_priority_field: fetch_fields.append(parent_priority_field)
    payload = {"jql": "key in (" + ",".join(sorted(want_keys)) + ")",
               "fields": fetch_fields, "maxResults": len(want_keys)}
    r = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
    if r.status_code == 200:
        data = r.json()
        by_key = {}
        for it in data.get("issues", []):
            k = it.get("key"); flds = it.get("fields") or {}
            meta = {"summary": flds.get("summary", "")}
            if parent_priority_field: meta["priority_num"] = flds.get(parent_priority_field)
            by_key[k] = meta
        for iss in issues:
            p = (iss.get("fields") or {}).get("parent")
            if isinstance(p, dict) and p.get("key"):
                meta = by_key.get(p["key"], {})
                p["_summary_cache"] = meta.get("summary", "")
                if "priority_num" in meta: p["_priority_num_cache"] = meta["priority_num"]

# ------------- Rendering, Teams posting, Main ------------
# (Due to space, leaving as skeleton; you'd integrate the rest of your rendering/build_grouped_markdown/post_to_teams here)

def main():
    base = env_or_config("JIRA_BASE_URL")
    email = env_or_config("JIRA_EMAIL")
    token = env_or_config("JIRA_API_TOKEN")
    jql = env_or_config("JIRA_JQL")
    fields_csv = os.getenv("JIRA_FIELDS")
    fields = [s.strip() for s in fields_csv.split(",")] if fields_csv else ["key","summary"]
    webhook = env_or_config("TEAMS_WEBHOOK_URL")

    debug_parent = parse_bool(env_or_config("DEBUG_PARENT_PRIORITY"), False)
    force_parent_get = parse_bool(env_or_config("PARENT_ENRICH_FORCE_GET"), False)

    print("jira_to_teams_report.py v2.5 (force-get option)")
    print(f"Querying Jira with JQL: {jql}")
    issues = fetch_enhanced(base, email, token, jql, fields, 100, 50, "", None)

    parent_rank_field = env_or_config("PARENT_RANK_FIELD") or ""
    epic_rank_field = env_or_config("PARENT_EPIC_RANK_FIELD") or ""
    parent_priority_field = env_or_config("PARENT_PRIORITY_FIELD") or ""

    bulk_fill_parent_summaries(base, email, token, issues, parent_rank_field, epic_rank_field,
                               parent_priority_field, debug=debug_parent, force_get=force_parent_get)

    print(f"Fetched {len(issues)} issues")
    # Continue with grouping + posting...

if __name__ == "__main__":
    main()
