#!/usr/bin/env python3
"""
jira_to_teams_report.py (parent grouping by Rank + child issue Rank sorting)

Key features
- Enhanced JQL search (POST /rest/api/3/search/jql) + legacy fallback
- Teams webhook posting (accept any 2xx)
- parentSummary enrichment (bulk one-call fetch when missing)
- Group-by-parent, groups ordered by **parent Rank (LexoRank)**; "No Parent" last
- **NEW:** Sort issues within each parent group by their own **issue Rank**

Env (required):
  JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_JQL, JIRA_FIELDS, TEAMS_WEBHOOK_URL

Env (optional):
  TITLE, DATE_FORMAT, TIMEZONE, MAX_ISSUES, BATCH_SIZE
  JIRA_USE_LEGACY="true"
  EXPAND="changelog,renderedFields"
  RECONCILE_ISSUE_IDS="10001,10042"
  GROUP_BY_PARENT="true"
  GROUP_STRIP_PARENT_SUMMARY="true"
  PARENT_RANK_FIELD="customfield_10019"   # parent Rank field (default Jira Cloud Rank)
  ISSUE_RANK_FIELD="customfield_10019"    # child issue Rank field (usually same as parent)
  SORT_CHILDREN_BY_RANK="true"            # sort issues within each parent by issue Rank
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
    "JIRA_FIELDS": ["key", "summary", "parentSummary", "status", "assignee", "updated"],
    "TEAMS_WEBHOOK_URL": "https://outlook.office.com/webhook/your-webhook-url",
    "TITLE": "Jira Report",
    "DATE_FORMAT": "%Y-%m-%d %H:%M",
    "TIMEZONE": "UTC",
    "MAX_ISSUES": 500,
    "BATCH_SIZE": 100,
    "JIRA_USE_LEGACY": "false",
    "EXPAND": "",
    "RECONCILE_ISSUE_IDS": "",
    "GROUP_BY_PARENT": "true",
    "GROUP_STRIP_PARENT_SUMMARY": "true",
    "PARENT_RANK_FIELD": "customfield_10019",
    "ISSUE_RANK_FIELD": "customfield_10019",
    "SORT_CHILDREN_BY_RANK": "true",
}

# ---------------- Helpers ----------------

def env_or_config(name: str, default=None):
    return os.getenv(name, CONFIG.get(name, default))

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
    """Prepare the fields array for Jira API request.
       - remove 'key' (it's top-level)
       - auto-include 'parent' when parentSummary is requested or when grouping by parent
       - auto-include issue Rank field if we intend to sort children by rank
    """
    req = [f for f in fields if f.lower() != "key"]

    group_by_parent = parse_bool(env_or_config("GROUP_BY_PARENT") or "true", default=True)
    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields) or group_by_parent:
        if "parent" not in {x.lower() for x in req}:
            req.append("parent")

    if parse_bool(env_or_config("SORT_CHILDREN_BY_RANK") or "true", default=True):
        issue_rank_field = env_or_config("ISSUE_RANK_FIELD") or "customfield_10019"
        if issue_rank_field not in req:
            req.append(issue_rank_field)

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

# ------------- Fetch (Enhanced + Legacy) -------------

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
            payload["expand"] = expand   # string (enhanced schema)
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

# ------------- Parent Summary + Rank Enrichment -------------

def bulk_fill_parent_summaries(base: str, email: str, token: str, issues: List[Dict[str, Any]], parent_rank_field: str) -> None:
    """Find parents without embedded summary and fetch their summaries + rank in one call.
       Injects _summary_cache and _rank_cache onto the parent objects.
    """
    want_keys = set()
    for iss in issues:
        fields = iss.get("fields") or {}
        parent = fields.get("parent")
        if isinstance(parent, dict):
            embedded = isinstance(parent.get("fields"), dict) and parent["fields"].get("summary")
            if not embedded and parent.get("key"):
                want_keys.add(parent["key"])

    if not want_keys:
        return

    jql = "key in (" + ",".join(sorted(want_keys)) + ")"
    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    payload = {
        "jql": jql,
        "fields": ["summary", parent_rank_field],
        "maxResults": len(want_keys),
    }
    try:
        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            return
        data = resp.json()
        by_key: Dict[str, Dict[str, Any]] = {}
        for it in data.get("issues", []):
            k = it.get("key")
            flds = it.get("fields") or {}
            by_key[k] = {
                "summary": flds.get("summary", ""),
                "rank": flds.get(parent_rank_field)
            }

        for iss in issues:
            fields = iss.get("fields") or {}
            parent = fields.get("parent")
            if isinstance(parent, dict) and parent.get("key"):
                meta = by_key.get(parent["key"], {})
                if isinstance(meta, dict):
                    if "summary" in meta:
                        parent["_summary_cache"] = meta.get("summary", parent.get("_summary_cache", ""))
                    if "rank" in meta:
                        parent["_rank_cache"] = meta.get("rank", parent.get("_rank_cache", None))
    except Exception:
        return

# ------------- Rendering (grouped; parent rank order; child rank order) -------------

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

def group_issues_by_parent(issues: List[Dict[str, Any]], issue_rank_field: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for iss in issues:
        f = iss.get("fields") or {}
        parent = f.get("parent")
        if isinstance(parent, dict):
            pkey = parent.get("key") or "NO_PARENT"
            # parent title
            if isinstance(parent.get("fields"), dict) and parent["fields"].get("summary"):
                ptitle = parent["fields"]["summary"]
            else:
                ptitle = parent.get("_summary_cache", "")
            # parent rank (try embedded default rank field or cached)
            pf = parent.get("fields") or {}
            prank = pf.get("customfield_10019") or parent.get("_rank_cache", "")
        else:
            pkey = "NO_PARENT"; ptitle = ""; prank = ""
        if pkey not in groups:
            groups[pkey] = {"title": ptitle, "rank": prank or "", "issues": []}
        if ptitle and not groups[pkey]["title"]:
            groups[pkey]["title"] = ptitle
        if prank and not groups[pkey].get("rank"):
            groups[pkey]["rank"] = prank
        # capture child rank for later sorting
        child_rank = (f.get(issue_rank_field) or "")
        iss["_child_rank_cache"] = child_rank
        groups[pkey]["issues"].append(iss)
    return groups

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

def build_grouped_markdown(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str,
                           strip_parent_summary: bool = True, issue_rank_field: str = "customfield_10019") -> str:
    if not issues:
        return "_No issues found for the given JQL._"
    groups = group_issues_by_parent(issues, issue_rank_field)
    fields_local = list(fields)
    if strip_parent_summary:
        fields_local = [f for f in fields_local if f.lower() not in {"parentsummary","parent_summary"}]

    def sort_group_key(k: str):
        # "NO_PARENT" groups last; otherwise by parent rank (lexicographically)
        if k == "NO_PARENT":
            return (1, "", k)
        rank = groups[k].get("rank", "")
        return (0, rank or "~", k)

    sections = []
    for pkey in sorted(groups.keys(), key=sort_group_key):
        meta = groups[pkey]
        title = meta.get("title") or ""
        if pkey == "NO_PARENT":
            heading = "**No Parent**"
        else:
            heading = f"**{pkey} — {title}**" if title else f"**{pkey}**"
        sections.append(heading)

        # Sort children by their rank (lexicographically); fallback by key
        children = list(meta["issues"])
        children.sort(key=lambda it: (it.get("_child_rank_cache") or "~", it.get("key") or ""))

        header = " | ".join(fields_local)
        sep = " | ".join(["---"] * len(fields_local))
        rows = [header, sep]
        for issue in children:
            vals = [extract_field(issue, f, tzname, datefmt).replace("\n", " ").strip() for f in fields_local]
            vals = [v.replace("|", "\\|") for v in vals]
            rows.append(" | ".join(vals))
        sections.append("\n".join(rows))
    return "\n\n".join(sections)

def chunk_text(text: str, max_len: int = 20000) -> List[str]:
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
    resp = requests.post(webhook_url, json=payload, timeout=45)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Teams webhook error {resp.status_code}: {resp.text}")

# ---------------- Main ----------------

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
    group_by_parent = parse_bool(env_or_config("GROUP_BY_PARENT") or "true", default=True)
    strip_parent_summary = parse_bool(env_or_config("GROUP_STRIP_PARENT_SUMMARY") or "true", default=True)
    parent_rank_field = env_or_config("PARENT_RANK_FIELD") or "customfield_10019"
    issue_rank_field = env_or_config("ISSUE_RANK_FIELD") or "customfield_10019"
    sort_children = parse_bool(env_or_config("SORT_CHILDREN_BY_RANK") or "true", default=True)

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

    # Enrich with parent summaries + rank if requested/needed
    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields) or group_by_parent:
        bulk_fill_parent_summaries(base, email, token, issues, parent_rank_field)

    print(f"Fetched {len(issues)} issues")
    if group_by_parent:
        markdown = build_grouped_markdown(issues, fields, tzname, datefmt,
                                          strip_parent_summary=strip_parent_summary,
                                          issue_rank_field=issue_rank_field if sort_children else "no_sort")
    else:
        markdown = build_markdown_table(issues, fields, tzname, datefmt)

    for i, chunk in enumerate(chunk_text(markdown), 1):
        chunk_title = title if i == 1 and len(markdown) == len(chunk) else f"{title} (part {i})"
        post_to_teams(webhook, chunk_title, chunk)
        print(f"Posted chunk {i} to Teams")

if __name__ == "__main__":
    main()
