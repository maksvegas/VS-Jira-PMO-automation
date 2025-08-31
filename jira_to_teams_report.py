#!/usr/bin/env python3
"""
jira_to_teams_report.py (grouped + parentSummary + safer Teams send)

Adds reliability features for Teams:
- TEAMS_MESSAGE_MODE: "card" (default) or "plain"
  * card: O365 MessageCard payload (legacy connector schema)
  * plain: {"text": "..."} minimal payload
- TEAMS_PROBE: "true" (default) -> sends a tiny probe message first
- TEAMS_CHUNK_LIMIT: character limit per chunk (default 10000)

Other features preserved:
- Enhanced JQL search + legacy fallback
- parentSummary enrichment
- Group-by-parent sections
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

# ----------------- (imports for types used in helpers) -----------------
from typing import List, Dict, Any  # re-import for static hints

# ------- Config Defaults (non-secret) -------
DEFAULTS = {
    "TITLE": "Jira Report",
    "DATE_FORMAT": "%Y-%m-%d %H:%M",
    "TIMEZONE": "UTC",
    "MAX_ISSUES": "500",
    "BATCH_SIZE": "100",
    "JIRA_USE_LEGACY": "false",
    "EXPAND": "",
    "RECONCILE_ISSUE_IDS": "",
    "GROUP_BY_PARENT": "true",
    "GROUP_STRIP_PARENT_SUMMARY": "true",
    # Teams reliability knobs
    "TEAMS_MESSAGE_MODE": "card",   # "card" or "plain"
    "TEAMS_PROBE": "true",
    "TEAMS_CHUNK_LIMIT": "10000",
}

def getenv(name: str, fallback_key: Optional[str] = None):
    if fallback_key is None:
        fallback_key = name
    return os.getenv(name, DEFAULTS.get(fallback_key))

def parse_bool(v: Optional[str], default=False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1","true","yes","y","on"}

def parse_reconcile_ids(csv: str) -> Optional[List[int]]:
    if not csv:
        return None
    ids = []
    for token in csv.split(","):
        t = token.strip()
        if t and t.isdigit():
            ids.append(int(t))
    return ids or None

def compute_request_fields(fields: List[str]) -> List[str]:
    req = [f for f in fields if f.lower() != "key"]
    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields):
        if "parent" not in {x.lower() for x in req}:
            req.append("parent")
    return req

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

# ----------- Fetchers -----------

def fetch_enhanced(base: str, email: str, token: str, jql: str, fields: List[str],
                   max_issues: int, batch_size: int, expand: str,
                   reconcile_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
    url = f"{base}/rest/api/3/search/jql"
    headers, auth = jira_auth(email, token)
    out: List[Dict[str, Any]] = []
    next_token: Optional[str] = None
    req_fields = compute_request_fields(fields)

    while True:
        remaining = max_issues - len(out)
        if remaining <= 0:
            break
        limit = min(batch_size, remaining)
        payload: Dict[str, Any] = {
            "jql": jql,
            "maxResults": limit,
            "fields": req_fields,
        }
        if expand:
            payload["expand"] = expand
        if reconcile_ids:
            payload["reconcileIssues"] = reconcile_ids
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
    req_fields = compute_request_fields(fields)

    while True:
        remaining = max_issues - len(out)
        if remaining <= 0:
            break
        limit = min(batch_size, remaining)
        payload: Dict[str, Any] = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": limit,
            "fields": req_fields,
        }
        if expand:
            payload["expand"] = [e.strip() for e in expand.split(",") if e.strip()]

        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        issues = data.get("issues", [])
        out.extend(issues)
        if not issues:
            break
        start_at += len(issues)
    return out

# ----------- Parent summaries -----------

def bulk_fill_parent_summaries(base: str, email: str, token: str, issues: List[Dict[str, Any]]) -> None:
    want_keys = set()
    for iss in issues:
        f = iss.get("fields") or {}
        parent = f.get("parent")
        if isinstance(parent, dict):
            embedded = isinstance(parent.get("fields"), dict) and parent["fields"].get("summary")
            if not embedded and parent.get("key"):
                want_keys.add(parent["key"])
    if not want_keys:
        return

    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    payload = {"jql": "key in (" + ",".join(sorted(want_keys)) + ")", "fields": ["summary"], "maxResults": len(want_keys)}
    try:
        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            return
        data = resp.json()
        by_key = {it.get("key"): (it.get("fields") or {}).get("summary", "") for it in data.get("issues", [])}
        for iss in issues:
            f = iss.get("fields") or {}
            parent = f.get("parent")
            if isinstance(parent, dict) and parent.get("key"):
                parent["_summary_cache"] = by_key.get(parent["key"], parent.get("_summary_cache", ""))
    except Exception:
        return

# ----------- Rendering -----------

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

    if f in {"parentsummary", "parent_summary"}:
        parent = fields.get("parent") or {}
        if isinstance(parent, dict):
            if isinstance(parent.get("fields"), dict) and parent["fields"].get("summary"):
                return parent["fields"]["summary"]
            return parent.get("_summary_cache", "")
        return ""

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

def group_issues_by_parent(issues: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for iss in issues:
        f = iss.get("fields") or {}
        parent = f.get("parent")
        if isinstance(parent, dict):
            pkey = parent.get("key") or "NO_PARENT"
            if isinstance(parent.get("fields"), dict) and parent["fields"].get("summary"):
                ptitle = parent["fields"]["summary"]
            else:
                ptitle = parent.get("_summary_cache", "")
        else:
            pkey = "NO_PARENT"; ptitle = ""
        if pkey not in groups:
            groups[pkey] = {"title": ptitle, "issues": []}
        if ptitle and not groups[pkey]["title"]:
            groups[pkey]["title"] = ptitle
        groups[pkey]["issues"].append(iss)
    return groups

def build_grouped_markdown(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str,
                           strip_parent_summary: bool = True) -> str:
    if not issues:
        return "_No issues found for the given JQL._"
    groups = group_issues_by_parent(issues)
    fields_local = list(fields)
    if strip_parent_summary:
        fields_local = [f for f in fields_local if f.lower() not in {"parentsummary","parent_summary"}]

    sections = []
    for pkey in sorted(groups.keys(), key=lambda k: (k == "NO_PARENT", k)):
        meta = groups[pkey]
        title = meta.get("title") or ""
        if pkey == "NO_PARENT":
            heading = "**No Parent**"
        else:
            heading = f"**{pkey} — {title}**" if title else f"**{pkey}**"
        sections.append(heading)

        header = " | ".join(fields_local)
        sep = " | ".join(["---"] * len(fields_local))
        rows = [header, sep]
        for issue in meta["issues"]:
            vals = [extract_field(issue, f, tzname, datefmt).replace("\n", " ").strip() for f in fields_local]
            vals = [v.replace("|", "\\|") for v in vals]
            rows.append(" | ".join(vals))
        sections.append("\n".join(rows))
    return "\n\n".join(sections)

def chunk_text(text: str, max_len: int) -> List[str]:
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

# ----------- Teams posting -----------

def post_to_teams(webhook_url: str, title: str, markdown_text: str, mode: str = "card") -> None:
    if mode == "plain":
        payload = {"text": f"**{title}**\n\n{markdown_text}"}
    else:
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": title,
            "themeColor": "0076D7",
            "title": title,
            "text": markdown_text
        }
    resp = requests.post(webhook_url, json=payload, timeout=45)
    print(f"Teams response: {resp.status_code}")
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Teams webhook error {resp.status_code}: {resp.text}")

# ---------------- Main ----------------

def main():
    base = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    jql = os.getenv("JIRA_JQL")
    fields_csv = os.getenv("JIRA_FIELDS")
    webhook = os.getenv("TEAMS_WEBHOOK_URL")

    title = getenv("TITLE")
    datefmt = getenv("DATE_FORMAT")
    tzname = getenv("TIMEZONE")
    max_issues = int(getenv("MAX_ISSUES"))
    batch_size = int(getenv("BATCH_SIZE"))
    use_legacy = parse_bool(getenv("JIRA_USE_LEGACY"), default=False)
    expand = getenv("EXPAND") or ""
    reconcile_ids = parse_reconcile_ids(getenv("RECONCILE_ISSUE_IDS") or "")
    group_by_parent = parse_bool(getenv("GROUP_BY_PARENT"), default=True)
    strip_parent_summary = parse_bool(getenv("GROUP_STRIP_PARENT_SUMMARY"), default=True)

    teams_mode = (getenv("TEAMS_MESSAGE_MODE") or "card").strip().lower()
    teams_probe = parse_bool(getenv("TEAMS_PROBE"), default=True)
    teams_chunk_limit = int(getenv("TEAMS_CHUNK_LIMIT"))

    missing = [n for n, v in [("JIRA_BASE_URL", base), ("JIRA_EMAIL", email), ("JIRA_API_TOKEN", token),
                               ("JIRA_JQL", jql), ("TEAMS_WEBHOOK_URL", webhook), ("JIRA_FIELDS", fields_csv)] if not v]
    if missing:
        print("Missing required configuration values:", ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    fields = [s.strip() for s in fields_csv.split(",")]

    print(f"Querying Jira with JQL: {jql}")
    if use_legacy:
        issues = fetch_legacy(base, email, token, jql, fields, max_issues, batch_size, expand)
    else:
        issues = fetch_enhanced(base, email, token, jql, fields, max_issues, batch_size, expand, reconcile_ids)

    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields):
        bulk_fill_parent_summaries(base, email, token, issues)

    print(f"Fetched {len(issues)} issues")
    body = build_grouped_markdown(issues, fields, tzname, datefmt, strip_parent_summary) if group_by_parent \
           else build_markdown_table(issues, fields, tzname, datefmt)

    # Probe first (helps catch bad webhook/channel setup)
    if teams_probe:
        post_to_teams(webhook, f"{title} (probe)", "Probe: webhook reachable ✅", mode=teams_mode)

    # Chunk and send
    for i, chunk in enumerate(chunk_text(body, teams_chunk_limit), 1):
        chunk_title = title if i == 1 and len(body) == len(chunk) else f"{title} (part {i})"
        post_to_teams(webhook, chunk_title, chunk, mode=teams_mode)
        print(f"Posted chunk {i} to Teams")

if __name__ == "__main__":
    main()
