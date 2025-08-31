#!/usr/bin/env python3
"""
jira_to_teams_report.py (auto-detect Rank + robust ordering)

New:
- AUTO_DETECT_RANK="true" (default): discovers the Rank field ID via GET /rest/api/3/field
  and uses it for both PARENT_RANK_FIELD and ISSUE_RANK_FIELD if you didn't set them.
- Fallback ordering: if a parent has no rank, we order that group by the **minimum child rank**
  (lexicographically). If children also lack ranks, we fall back to parent key.
- DEBUG_ORDER prints which rank field IDs are used and the computed order basis.

Existing:
- Enhanced JQL search + legacy fallback
- Teams webhook post (accepts any 2xx)
- parentSummary enrichment
- Group-by-parent; issues within parent sorted by issue rank
"""

import os, sys, json
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
    "PARENT_RANK_FIELD": "",           # left blank on purpose to allow auto-detect
    "ISSUE_RANK_FIELD": "",
    "SORT_CHILDREN_BY_RANK": "true",
    "DEBUG_ORDER": "false",
    "AUTO_DETECT_RANK": "true",
}

def env_or_config(name: str, default=None):
    return os.getenv(name, CONFIG.get(name, default))

def parse_bool(v: Optional[str], default=False) -> bool:
    if v is None: return default
    return v.strip().lower() in {"1","true","yes","y","on"}

def parse_reconcile_ids(csv: str) -> Optional[List[int]]:
    if not csv: return None
    ids = []
    for t in csv.split(","):
        t = t.strip()
        if t.isdigit(): ids.append(int(t))
    return ids or None

def jira_auth(email: str, token: str):
    return {"Accept":"application/json","Content-Type":"application/json"}, (email, token)

def detect_rank_field(base: str, email: str, token: str) -> Optional[str]:
    """Return customfield id for Rank by inspecting /field metadata."""
    try:
        url = f"{base}/rest/api/3/field"
        headers, auth = jira_auth(email, token)
        r = requests.get(url, headers=headers, auth=auth, timeout=60)
        if r.status_code != 200:
            return None
        for f in r.json():
            name = (f.get("name") or "").strip().lower()
            fid = f.get("id") or ""
            schema = f.get("schema") or {}
            t = (schema.get("type") or "").lower()
            c = (schema.get("custom") or "").lower()
            # Heuristics: field named "Rank" with type 'string' and custom key contains 'rank'
            if name == "rank" and (t == "string" or "rank" in c or "rank" in fid.lower() or "lexorank" in c):
                return fid
        # fallback: first field with custom schema containing "rank"
        for f in r.json():
            fid = f.get("id") or ""
            schema = f.get("schema") or {}
            c = (schema.get("custom") or "").lower()
            if "rank" in c or "lexorank" in c:
                return fid
    except Exception:
        return None
    return None

def compute_request_fields(fields: List[str], issue_rank_field: str, group_by_parent: bool) -> List[str]:
    req = [f for f in fields if f.lower() != "key"]
    if any(f.lower() in {"parentsummary","parent_summary"} for f in fields) or group_by_parent:
        if "parent" not in {x.lower() for x in req}: req.append("parent")
    if issue_rank_field and issue_rank_field not in req:
        req.append(issue_rank_field)
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

# --------- Fetchers ---------

def fetch_enhanced(base, email, token, jql, req_fields, max_issues, batch_size, expand, reconcile_ids):
    url = f"{base}/rest/api/3/search/jql"
    headers, auth = jira_auth(email, token)
    out, next_token = [], None
    while True:
        remaining = max_issues - len(out)
        if remaining <= 0: break
        limit = min(batch_size, remaining)
        payload = {"jql": jql, "maxResults": limit, "fields": req_fields}
        if expand: payload["expand"] = expand
        if reconcile_ids: payload["reconcileIssues"] = reconcile_ids
        if next_token: payload["nextPageToken"] = next_token
        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        out.extend(data.get("issues", []))
        next_token = data.get("nextPageToken")
        if not next_token or not data.get("issues"): break
    return out

def fetch_legacy(base, email, token, jql, req_fields, max_issues, batch_size, expand):
    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    out, start_at = [], 0
    while True:
        remaining = max_issues - len(out)
        if remaining <= 0: break
        limit = min(batch_size, remaining)
        payload = {"jql": jql, "startAt": start_at, "maxResults": limit, "fields": req_fields}
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

# --------- Enrichment ---------

def bulk_fill_parent_summaries(base, email, token, issues, parent_rank_field):
    want_keys = set()
    for iss in issues:
        f = iss.get("fields") or {}
        parent = f.get("parent")
        if isinstance(parent, dict):
            embedded = isinstance(parent.get("fields"), dict) and parent["fields"].get("summary")
            if not embedded and parent.get("key"): want_keys.add(parent["key"])
    if not want_keys: return

    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    payload = {"jql": "key in (" + ",".join(sorted(want_keys)) + ")", "fields": ["summary", parent_rank_field], "maxResults": len(want_keys)}
    try:
        r = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if r.status_code != 200: return
        data = r.json()
        by_key = {}
        for it in data.get("issues", []):
            k = it.get("key"); flds = it.get("fields") or {}
            by_key[k] = {"summary": flds.get("summary",""), "rank": flds.get(parent_rank_field)}
        for iss in issues:
            parent = (iss.get("fields") or {}).get("parent")
            if isinstance(parent, dict) and parent.get("key"):
                meta = by_key.get(parent["key"], {})
                parent["_summary_cache"] = meta.get("summary", parent.get("_summary_cache",""))
                parent["_rank_cache"] = meta.get("rank", parent.get("_rank_cache", None))
    except Exception:
        return

# --------- Rendering ---------

def extract_field(issue, field, tzname, datefmt):
    f = field.lower()
    if f == "key": return issue.get("key","")
    fields = issue.get("fields") or {}
    if f == "summary": return fields.get("summary","") or ""
    if f == "status":
        st = fields.get("status") or {}
        return (st.get("name") or st.get("statusCategory",{}).get("name") or "") or ""
    if f == "assignee":
        asg = fields.get("assignee") or {}
        return asg.get("displayName") or asg.get("name") or ""
    if f in {"updated","created","duedate","resolutiondate"}:
        v = fields.get(field) or fields.get(f); return format_date(v, tzname, datefmt)
    if f in {"parentsummary","parent_summary"}:
        parent = fields.get("parent") or {}
        if isinstance(parent, dict):
            if isinstance(parent.get("fields"), dict) and parent["fields"].get("summary"):
                return parent["fields"]["summary"]
            return parent.get("_summary_cache","")
        return ""
    val = fields.get(field) or fields.get(f)
    if isinstance(val, dict):
        for k in ("displayName","name","value","id"):
            if k in val and isinstance(val[k], (str,int)): return str(val[k])
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

def group_issues_by_parent(issues, issue_rank_field, parent_rank_field):
    groups: Dict[str, Dict[str, Any]] = {}
    for iss in issues:
        f = iss.get("fields") or {}
        parent = f.get("parent")
        if isinstance(parent, dict):
            pkey = parent.get("key") or "NO_PARENT"
            pf = parent.get("fields") or {}
            ptitle = (pf.get("summary") or parent.get("_summary_cache","") or "")
            prank = pf.get(parent_rank_field) or parent.get("_rank_cache","") or ""
        else:
            pkey = "NO_PARENT"; ptitle = ""; prank = ""
        if pkey not in groups:
            groups[pkey] = {"title": ptitle, "rank": prank, "issues": []}
        if ptitle and not groups[pkey]["title"]:
            groups[pkey]["title"] = ptitle
        if prank and not groups[pkey].get("rank"):
            groups[pkey]["rank"] = prank
        iss["_child_rank_cache"] = (f.get(issue_rank_field) or "")
        groups[pkey]["issues"].append(iss)
    # Pre-compute min child rank for fallback ordering
    for k, meta in groups.items():
        child_ranks = [i.get("_child_rank_cache") for i in meta["issues"] if i.get("_child_rank_cache")]
        meta["_min_child_rank"] = min(child_ranks) if child_ranks else ""
    return groups

def build_markdown_table(issues, fields, tzname, datefmt):
    if not issues: return "_No issues found for the given JQL._"
    header = " | ".join(fields)
    sep = " | ".join(["---"] * len(fields))
    rows = [header, sep]
    for issue in issues:
        vals = [extract_field(issue, f, tzname, datefmt).replace("\n"," ").strip() for f in fields]
        vals = [v.replace("|","\\|") for v in vals]
        rows.append(" | ".join(vals))
    return "\n".join(rows)

def build_grouped_markdown(issues, fields, tzname, datefmt, strip_parent_summary, issue_rank_field, parent_rank_field, debug=False):
    if not issues: return "_No issues found for the given JQL._"
    groups = group_issues_by_parent(issues, issue_rank_field, parent_rank_field)
    fields_local = [f for f in fields if not(strip_parent_summary and f.lower() in {"parentsummary","parent_summary"})]

    def sort_group_key(k: str):
        if k == "NO_PARENT": return (1, "", "", k)
        prank = groups[k].get("rank","")
        min_child = groups[k].get("_min_child_rank","")
        # Prefer parent rank, otherwise min child rank; fallback key
        basis = prank or min_child or "~"
        return (0, basis, k)

    ordered = sorted(groups.keys(), key=sort_group_key)

    if debug:
        print(f"Using parent_rank_field={parent_rank_field} issue_rank_field={issue_rank_field}")
        print("Parent group order:")
        for k in ordered:
            meta = groups[k]
            print(f"  {k}: prank={meta.get('rank','')} min_child={meta.get('_min_child_rank','')} title={meta.get('title','')} size={len(meta['issues'])}")
        if ordered:
            first = ordered[0]
            sample = [i.get('_child_rank_cache') for i in groups[first]['issues'][:5]]
            print(f"Sample child ranks for {first}: {sample}")

    sections = []
    for pkey in ordered:
        meta = groups[pkey]
        title = meta.get("title") or ""
        heading = "**No Parent**" if pkey == "NO_PARENT" else f"**{pkey} — {title}**" if title else f"**{pkey}**"
        sections.append(heading)

        children = list(meta["issues"])
        children.sort(key=lambda it: (it.get("_child_rank_cache") or "~", it.get("key") or ""))

        header = " | ".join(fields_local)
        sep = " | ".join(["---"] * len(fields_local))
        rows = [header, sep]
        for issue in children:
            vals = [extract_field(issue, f, tzname, datefmt).replace("\n"," ").strip() for f in fields_local]
            vals = [v.replace("|","\\|") for v in vals]
            rows.append(" | ".join(vals))
        sections.append("\n".join(rows))
    return "\n".join(sections)

def chunk_text(text: str, max_len: int = 20000):
    if len(text) <= max_len: return [text]
    chunks, cur, n = [], [], 0
    for line in text.splitlines(False):
        if n + len(line) + 1 > max_len:
            chunks.append("\n".join(cur)); cur, n = [line], len(line) + 1
        else:
            cur.append(line); n += len(line) + 1
    if cur: chunks.append("\n".join(cur))
    return chunks

def post_to_teams(webhook_url: str, title: str, markdown_text: str):
    payload = {"@type": "MessageCard", "@context": "http://schema.org/extensions",
               "summary": title, "themeColor": "0076D7", "title": title, "text": markdown_text}
    resp = requests.post(webhook_url, json=payload, timeout=45)
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
    group_by_parent = parse_bool(env_or_config("GROUP_BY_PARENT") or "true", default=True)
    strip_parent_summary = parse_bool(env_or_config("GROUP_STRIP_PARENT_SUMMARY") or "true", default=True)
    parent_rank_field = env_or_config("PARENT_RANK_FIELD") or ""
    issue_rank_field = env_or_config("ISSUE_RANK_FIELD") or ""
    sort_children = parse_bool(env_or_config("SORT_CHILDREN_BY_RANK") or "true", default=True)
    debug = parse_bool(env_or_config("DEBUG_ORDER") or "false", default=False)
    auto_detect = parse_bool(env_or_config("AUTO_DETECT_RANK") or "true", default=True)

    missing = [n for n, v in [("JIRA_BASE_URL", base), ("JIRA_EMAIL", email), ("JIRA_API_TOKEN", token),
                               ("JIRA_JQL", jql), ("TEAMS_WEBHOOK_URL", webhook)] if not v]
    if missing:
        print("Missing required configuration values:", ", ".join(missing), file=sys.stderr); sys.exit(2)

    # Auto-detect Rank field if not provided
    if auto_detect and (not parent_rank_field or not issue_rank_field):
        fid = detect_rank_field(base, email, token)
        if fid:
            if not parent_rank_field: parent_rank_field = fid
            if not issue_rank_field: issue_rank_field = fid
            if debug: print(f"Auto-detected Rank field id: {fid}")
        else:
            if debug: print("Warning: could not auto-detect Rank field id.")

    req_fields = compute_request_fields(fields, issue_rank_field if sort_children else "", group_by_parent)

    print(f"Querying Jira with JQL: {jql}")
    if use_legacy:
        issues = fetch_legacy(base, email, token, jql, req_fields, max_issues, batch_size, expand)
    else:
        issues = fetch_enhanced(base, email, token, jql, req_fields, max_issues, batch_size, expand, reconcile_ids)

    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields) or group_by_parent:
        # Ensure we have a parent rank field to ask for; if not, detection failed — grouping will fall back to child min rank
        if not parent_rank_field:
            if debug: print("Note: parent rank field id empty; will use min child rank as fallback ordering.")
        else:
            bulk_fill_parent_summaries(base, email, token, issues, parent_rank_field)

    print(f"Fetched {len(issues)} issues")
    if group_by_parent:
        markdown = build_grouped_markdown(
            issues, fields, tzname, datefmt,
            strip_parent_summary=strip_parent_summary,
            issue_rank_field=(issue_rank_field if sort_children else ""),
            parent_rank_field=(parent_rank_field or ""),
            debug=debug
        )
    else:
        markdown = build_markdown_table(issues, fields, tzname, datefmt)

    for i, chunk in enumerate(chunk_text(markdown), 1):
        chunk_title = title if i == 1 and len(markdown) == len(chunk) else f"{title} (part {i})"
        post_to_teams(webhook, chunk_title, chunk)
        print(f"Posted chunk {i} to Teams")

if __name__ == "__main__":
    main()
