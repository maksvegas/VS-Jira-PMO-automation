#!/usr/bin/env python3
"""
jira_to_teams_report.py (enhanced-compatible)

- Supports Jira enhanced search: POST /rest/api/3/search/jql
  * Pagination via nextPageToken
  * expand is a comma-separated string
  * reconcileIssues is an array of numeric issue IDs
- Backward-compatible with legacy POST /rest/api/3/search
  * Pagination via startAt
  * expand is an array of strings

Env vars:
  JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_JQL, JIRA_FIELDS, TEAMS_WEBHOOK_URL
  TITLE, DATE_FORMAT, TIMEZONE, MAX_ISSUES, BATCH_SIZE
  JIRA_USE_LEGACY             -> "true" to force legacy endpoint (default false)
  EXPAND                      -> e.g. "changelog,renderedFields"
  RECONCILE_ISSUE_IDS         -> CSV of numeric issue IDs for strong consistency (enhanced only)
"""
import os
import sys
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

CONFIG = {
    "JIRA_BASE_URL": "https://your-domain.atlassian.net",
    "JIRA_EMAIL": "you@example.com",
    "JIRA_API_TOKEN": "YOUR_API_TOKEN",
    "JIRA_JQL": "project = ABC ORDER BY updated DESC",
    "JIRA_FIELDS": ["key", "summary", "status", "assignee", "updated"],
    "TEAMS_WEBHOOK_URL": "https://outlook.office.com/webhook/your-webhook-url",
    "TITLE": "Jira Report",
    "DATE_FORMAT": "%Y-%m-%d %H:%M",
    "TIMEZONE": "UTC",
    "MAX_ISSUES": 500,
    "BATCH_SIZE": 100,
    "JIRA_USE_LEGACY": "false",
    "EXPAND": "",
    "RECONCILE_ISSUE_IDS": "",
}

def env_or_config(name: str, default=None):
    return os.getenv(name, CONFIG.get(name, default))

def parse_bool(v: Optional[str], default=False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1","true","yes","y","on"}

def normalize_fields(fields: List[str]) -> List[str]:
    return [f for f in fields if f.lower() != "key"]

def parse_reconcile_ids(csv: str) -> Optional[List[int]]:
    if not csv:
        return None
    ids = []
    for token in csv.split(","):
        t = token.strip()
        if not t:
            continue
        if t.isdigit():
            ids.append(int(t))
        else:
            # Non-numeric (likely keys). Enhanced API expects numeric IDs.
            # We silently ignore non-digit tokens to avoid 400s.
            continue
    return ids or None

def format_date(dt_str: str, tzname: str, fmt: str) -> str:
    if not dt_str:
        return ""
    try:
        s = dt_str.replace("+0000", "+00:00")
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if tzname and ZoneInfo:
            dt = dt.astimezone(ZoneInfo(tzname))
        return dt.strftime(fmt)
    except Exception:
        return dt_str

def jira_auth(email: str, token: str):
    return {"Accept": "application/json", "Content-Type": "application/json"}, (email, token)

def fetch_enhanced(base: str, email: str, token: str, jql: str, fields: List[str],
                   max_issues: int, batch_size: int, expand: str,
                   reconcile_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
    url = f"{base}/rest/api/3/search/jql"
    headers, auth = jira_auth(email, token)
    out: List[Dict[str, Any]] = []
    next_token: Optional[str] = None

    while True:
        remaining = max_issues - len(out)
        if remaining <= 0:
            break
        limit = min(batch_size, remaining)
        payload: Dict[str, Any] = {
            "jql": jql,
            "maxResults": limit,
            "fields": normalize_fields(fields),
        }
        if expand:
            payload["expand"] = expand   # string per enhanced schema
        if reconcile_ids:
            payload["reconcileIssues"] = reconcile_ids  # array<int>
        if next_token:
            payload["nextPageToken"] = next_token

        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        issues = data.get("issues", [])
        out.extend(issues)
        next_token = data.get("nextPageToken")
        if not next_token or not issues or len(out) >= max_issues:
            break
    return out

def fetch_legacy(base: str, email: str, token: str, jql: str, fields: List[str],
                 max_issues: int, batch_size: int, expand: str) -> List[Dict[str, Any]]:
    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    out: List[Dict[str, Any]] = []
    start_at = 0

    while True:
        remaining = max_issues - len(out)
        if remaining <= 0:
            break
        limit = min(batch_size, remaining)
        payload: Dict[str, Any] = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": limit,
            "fields": normalize_fields(fields),
        }
        if expand:
            payload["expand"] = [e.strip() for e in expand.split(",") if e.strip()]  # array per legacy schema

        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        issues = data.get("issues", [])
        out.extend(issues)
        if not issues or len(issues) == 0:
            break
        start_at += len(issues)
    return out

def extract_field(issue: Dict[str, Any], field: str, tzname: str, datefmt: str) -> str:
    f = field.lower()
    if f == "key":
        return issue.get("key", "")
    fields = issue.get("fields", {})

    if f == "summary":
        return fields.get("summary", "") or ""
    if f == "status":
        st = fields.get("status") or {}
        return (st.get("name") or st.get("statusCategory", {}).get("name") or "") or ""
    if f == "assignee":
        asg = fields.get("assignee") or {}
        return asg.get("displayName") or asg.get("name") or ""
    if f in {"updated", "created", "duedate", "resolutiondate"}:
        return format_date(fields.get(field) or fields.get(f), tzname, datefmt)

    val = fields.get(field) or fields.get(f)
    if isinstance(val, dict):
        for k in ("displayName", "name", "value", "id"):
            if k in val and isinstance(val[k], (str, int)):
                return str(val[k])
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, list):
        simple = []
        for item in val:
            if isinstance(item, dict):
                simple.append(item.get("name") or item.get("value") or item.get("displayName") or item.get("key") or str(item))
            else:
                simple.append(str(item))
        return ", ".join([s for s in simple if s])
    return "" if val is None else str(val)

def build_markdown_table(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str) -> str:
    if not issues:
        return "_No issues found for the given JQL._"
    header = " | ".join(fields)
    sep = " | ".join(["---"] * len(fields))
    rows = [header, sep]
    for issue in issues:
        vals = [extract_field(issue, f, tzname, datefmt).replace("\n", " ").strip() for f in fields]
        vals = [v.replace("|", "\\|") for v in vals]
        rows.append(" | ".join(vals))
    return "\n".join(rows)

def chunk_text(text: str, max_len: int = 25000):
    if len(text) <= max_len:
        return [text]
    chunks, current, n = [], [], 0
    for line in text.splitlines(False):
        if n + len(line) + 1 > max_len:
            chunks.append("\n".join(current)); current, n = [line], len(line) + 1
        else:
            current.append(line); n += len(line) + 1
    if current: chunks.append("\n".join(current))
    return chunks

def post_to_teams(webhook_url: str, title: str, markdown_text: str) -> None:
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": title,
        "themeColor": "0076D7",
        "title": title,
        "text": markdown_text
    }
    resp = requests.post(webhook_url, json=payload, timeout=30)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Teams webhook error {resp.status_code}: {resp.text}")

def main():
    base = env_or_config("JIRA_BASE_URL")
    email = env_or_config("JIRA_EMAIL")
    token = env_or_config("JIRA_API_TOKEN")
    jql = env_or_config("JIRA_JQL")
    fields_csv = os.getenv("JIRA_FIELDS")
    fields = [s.strip() for s in fields_csv.split(",")] if fields_csv else CONFIG["JIRA_FIELDS"]
    webhook = env_or_config("TEAMS_WEBHOOK_URL")
    title = env_or_config("TITLE") or CONFIG["TITLE"]
    datefmt = env_or_config("DATE_FORMAT") or CONFIG["DATE_FORMAT"]
    tzname = env_or_config("TIMEZONE") or CONFIG["TIMEZONE"]
    max_issues = int(env_or_config("MAX_ISSUES") or CONFIG["MAX_ISSUES"])
    batch_size = int(env_or_config("BATCH_SIZE") or CONFIG["BATCH_SIZE"])
    use_legacy = parse_bool(env_or_config("JIRA_USE_LEGACY"), default=False)
    expand = env_or_config("EXPAND") or ""
    reconcile_ids = parse_reconcile_ids(env_or_config("RECONCILE_ISSUE_IDS") or "")

    missing = [n for n, v in [("JIRA_BASE_URL", base), ("JIRA_EMAIL", email), ("JIRA_API_TOKEN", token),
                               ("JIRA_JQL", jql), ("TEAMS_WEBHOOK_URL", webhook)] if not v]
    if missing:
        print("Missing required configuration values:", ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    print(f"Querying Jira with JQL: {jql}")
    if use_legacy:
        issues = fetch_legacy(base, email, token, jql, fields, max_issues, batch_size, expand)
    else:
        issues = fetch_enhanced(base, email, token, jql, fields, max_issues, batch_size, expand, reconcile_ids)

    print(f"Fetched {len(issues)} issues")
    markdown_table = build_markdown_table(issues, fields, tzname, datefmt)

    for i, chunk in enumerate(chunk_text(markdown_table), 1):
        chunk_title = title if i == 1 and len(markdown_table) == len(chunk) else f"{title} (part {i})"
        post_to_teams(webhook, chunk_title, chunk)
        print(f"Posted chunk {i} to Teams")

if __name__ == "__main__":
    main()
