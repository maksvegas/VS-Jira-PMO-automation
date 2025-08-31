#!/usr/bin/env python3
"""
jira_to_teams_report.py — grouped Jira → Teams report
Ordering: explicit > numeric priority > parent rank/epic rank > min child rank > key

Features
- Enhanced JQL search (POST /rest/api/3/search/jql) with legacy fallback
- Teams MessageCard posting (accepts any 2xx)
- Parent enrichment in one call: summary + Rank + Epic Rank + numeric priority
- Group by parent; children sorted inside each parent
- Sorting controls:
  • PARENT_ORDER (explicit order by keys/titles)
  • Numeric parent priority (e.g., "Priority Rank" custom field)
  • Parent rank / Epic rank
  • Min child rank as fallback
  • Configurable directions for parent rank & child rank, and numeric priority
- Debug output to verify chosen order basis

Env (required)
  JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_JQL, JIRA_FIELDS, TEAMS_WEBHOOK_URL

Key Env (optional)
  TITLE, DATE_FORMAT, TIMEZONE, MAX_ISSUES, BATCH_SIZE
  JIRA_USE_LEGACY="true"
  EXPAND="changelog,renderedFields"
  RECONCILE_ISSUE_IDS="10001,10042"
  GROUP_BY_PARENT="true"
  GROUP_STRIP_PARENT_SUMMARY="true"

  # Rank
  AUTO_DETECT_RANK="true"
  AUTO_DETECT_EPIC_RANK="true"
  PARENT_RANK_FIELD="customfield_10015"
  PARENT_EPIC_RANK_FIELD=""
  ISSUE_RANK_FIELD="customfield_10015"
  PARENT_RANK_DIRECTION="desc"    # top-of-board first
  CHILD_RANK_DIRECTION="asc"

  # Numeric priority fields
  PARENT_PRIORITY_FIELD="customfield_10487"   # e.g., "Priority Rank" (numeric)
  PARENT_PRIORITY_DIRECTION="asc"             # 1,2,3 = high→low
  CHILD_PRIORITY_FIELD=""
  CHILD_PRIORITY_DIRECTION="asc"

  # Explicit parent override
  PARENT_ORDER=""                              # CSV of keys or titles
  PARENT_ORDER_MODE="auto"                     # auto|keys|titles

  # Debug
  DEBUG_ORDER="false"
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
    # Rank defaults
    "PARENT_RANK_FIELD": "",
    "PARENT_EPIC_RANK_FIELD": "",
    "ISSUE_RANK_FIELD": "",
    "AUTO_DETECT_RANK": "true",
    "AUTO_DETECT_EPIC_RANK": "true",
    "USE_EPIC_RANK_FOR_EPICS": "true",
    "PARENT_RANK_DIRECTION": "desc",
    "CHILD_RANK_DIRECTION": "asc",
    # Priority defaults
    "PARENT_PRIORITY_FIELD": "",
    "PARENT_PRIORITY_DIRECTION": "asc",
    "CHILD_PRIORITY_FIELD": "",
    "CHILD_PRIORITY_DIRECTION": "asc",
    # Explicit override
    "PARENT_ORDER": "",
    "PARENT_ORDER_MODE": "auto",
    # Debug
    "DEBUG_ORDER": "false",
}

# ---------------- Helpers ----------------

def env_or_config(name: str, default=None):
    return os.getenv(name, CONFIG.get(name, default))

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
        # Rank
        for f in arr:
            name = (f.get("name") or "").strip().lower()
            fid = f.get("id") or ""
            schema = f.get("schema") or {}
            t = (schema.get("type") or "").lower()
            c = (schema.get("custom") or "").lower()
            if name == "rank" and (t == "string" or "rank" in c or "lexorank" in c or "rank" in fid.lower()):
                found["rank"] = fid; break
        if not found["rank"]:
            for f in arr:
                fid = f.get("id") or ""
                c = ((f.get("schema") or {}).get("custom") or "").lower()
                if "lexorank" in c or "rank" in c:
                    found["rank"] = fid; break
        # Epic Rank
        for f in arr:
            name = (f.get("name") or "").strip().lower()
            fid = f.get("id") or ""
            schema = f.get("schema") or {}
            c = (schema.get("custom") or "").lower()
            if "epic rank" in name or "epic-rank" in c or "epicrank" in c:
                found["epic_rank"] = fid; break
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

# ------------- Fetch -------------

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

# ------------- Enrichment -------------

def bulk_fill_parent_summaries(base: str, email: str, token: str, issues: List[Dict[str, Any]],
                               parent_rank_field: str, epic_rank_field: str, parent_priority_field: str = "") -> None:
    """Fetch summary + rank(s) + numeric priority for **all** parents referenced by the issues."""
    want_keys = set()
    for iss in issues:
        fields = iss.get("fields") or {}
        parent = fields.get("parent")
        if isinstance(parent, dict) and parent.get("key"):
            want_keys.add(parent["key"])
    if not want_keys: return
    url = f"{base}/rest/api/3/search"
    headers, auth = jira_auth(email, token)
    fetch_fields = ["summary"]
    if parent_rank_field: fetch_fields.append(parent_rank_field)
    if epic_rank_field and epic_rank_field not in fetch_fields: fetch_fields.append(epic_rank_field)
    if parent_priority_field and parent_priority_field not in fetch_fields: fetch_fields.append(parent_priority_field)
    payload = {"jql": "key in (" + ",".join(sorted(want_keys)) + ")",
               "fields": fetch_fields, "maxResults": len(want_keys)}
    try:
        r = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if r.status_code != 200: return
        data = r.json()
        by_key: Dict[str, Dict[str, Any]] = {}
        for it in data.get("issues", []):
            k = it.get("key"); flds = it.get("fields") or {}
            meta = {"summary": flds.get("summary", "")}
            if parent_rank_field: meta["rank"] = flds.get(parent_rank_field)
            if epic_rank_field:   meta["epic_rank"] = flds.get(epic_rank_field)
            if parent_priority_field: meta["priority_num"] = flds.get(parent_priority_field)
            by_key[k] = meta
        for iss in issues:
            fields = iss.get("fields") or {}
            parent = fields.get("parent")
            if isinstance(parent, dict) and parent.get("key"):
                meta = by_key.get(parent["key"], {})
                parent["_summary_cache"] = meta.get("summary", parent.get("_summary_cache", ""))
                if "rank" in meta: parent["_rank_cache"] = meta.get("rank", parent.get("_rank_cache", None))
                if "epic_rank" in meta: parent["_epic_rank_cache"] = meta.get("epic_rank", parent.get("_epic_rank_cache", None))
                if "priority_num" in meta: parent["_priority_num_cache"] = meta.get("priority_num", parent.get("_priority_num_cache", None))
    except Exception:
        return

# ------------- Rendering -------------

def extract_field(issue: Dict[str, Any], field: str, tzname: str, datefmt: str) -> str:
    f = field.lower()
    if f == "key": return issue.get("key", "")
    fields = issue.get("fields", {})

    if f == "summary": return fields.get("summary", "") or ""
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

def group_issues_by_parent(issues: List[Dict[str, Any]], issue_rank_field: str,
                           parent_rank_field: str, epic_rank_field: str, use_epic_rank: bool,
                           parent_priority_field: str = "", child_priority_field: str = "") -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for iss in issues:
        f = iss.get("fields") or {}
        parent = f.get("parent")
        if isinstance(parent, dict):
            pkey = parent.get("key") or "NO_PARENT"
            pf = parent.get("fields") or {}
            ptitle = (pf.get("summary") or parent.get("_summary_cache", "") or "")
            ptype = (pf.get("issuetype") or {}).get("name") if isinstance(pf.get("issuetype"), dict) else ""
            # numeric priority (parent)
            pprio_val = None
            if parent_priority_field:
                pprio_val = pf.get(parent_priority_field)
                if pprio_val is None:
                    pprio_val = parent.get("_priority_num_cache", None)
                try:
                    pprio_val = float(pprio_val) if pprio_val is not None and pprio_val != "" else None
                except Exception:
                    pprio_val = None
            # rank sources
            prank_main = (pf.get(parent_rank_field) if parent_rank_field else None) or parent.get("_rank_cache", "") or ""
            prank_epic = (pf.get(epic_rank_field) if epic_rank_field else None) or parent.get("_epic_rank_cache", "") or ""
            prank = ""
            prank_source = ""
            if use_epic_rank and str(ptype).lower() == "epic" and prank_epic:
                prank = prank_epic; prank_source = "epic"
            else:
                prank = prank_main or prank_epic
                prank_source = "rank" if prank_main else ("epic" if prank_epic else "")
        else:
            pkey = "NO_PARENT"; ptitle = ""; prank = ""; prank_source = ""; pprio_val = None
        if pkey not in groups:
            groups[pkey] = {"title": ptitle, "rank": prank or "", "rank_source": prank_source, "priority": pprio_val, "issues": []}
        if ptitle and not groups[pkey]["title"]:
            groups[pkey]["title"] = ptitle
        if prank and not groups[pkey].get("rank"):
            groups[pkey]["rank"] = prank
            groups[pkey]["rank_source"] = prank_source or groups[pkey].get("rank_source","")
        if pprio_val is not None and groups[pkey].get("priority") is None:
            groups[pkey]["priority"] = pprio_val
        # child rank + priority
        child_rank = (f.get(issue_rank_field) or "")
        iss["_child_rank_cache"] = child_rank
        if child_priority_field:
            cprio = f.get(child_priority_field)
            try:
                cprio = float(cprio) if cprio is not None and cprio != "" else None
            except Exception:
                cprio = None
            iss["_child_priority_cache"] = cprio
        groups[pkey]["issues"].append(iss)
    # min child ranks for fallback
    for k, meta in groups.items():
        cr = [i.get("_child_rank_cache") for i in meta["issues"] if i.get("_child_rank_cache")]
        meta["_min_child_rank"] = min(cr) if cr else ""
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
                           strip_parent_summary: bool, issue_rank_field: str,
                           parent_rank_field: str, epic_rank_field: str, use_epic_rank: bool,
                           parent_dir: str = "desc", child_dir: str = "asc",
                           parent_order: str = "", parent_order_mode: str = "auto",
                           parent_priority_field: str = "", child_priority_field: str = "",
                           parent_priority_dir: str = "asc", child_priority_dir: str = "asc",
                           debug=False) -> str:
    if not issues:
        return "_No issues found for the given JQL._"
    groups = group_issues_by_parent(issues, issue_rank_field, parent_rank_field, epic_rank_field, use_epic_rank,
                                    parent_priority_field=parent_priority_field, child_priority_field=child_priority_field)

    # explicit order map
    order_map = {}
    if parent_order:
        raw_items = [x.strip() for x in parent_order.split(",") if x.strip()]
        for idx, item in enumerate(raw_items):
            order_map[item.lower()] = idx

    fields_local = list(fields)
    if strip_parent_summary:
        fields_local = [f for f in fields_local if f.lower() not in {"parentsummary","parent_summary"}]

    def explicit_index(k: str):
        meta = groups[k]
        title = (meta.get("title") or "").lower()
        if parent_order_mode in ("auto","keys"):
            if k.lower() in order_map: return order_map[k.lower()]
        if parent_order_mode in ("auto","titles"):
            if title and title.lower() in order_map: return order_map[title.lower()]
        return None

    def sort_group_key(k: str):
        if k == "NO_PARENT": return (9999, 0, "", k)  # always last unless explicitly listed
        idx = explicit_index(k)
        if idx is not None:
            return (0, idx, "", k)
        # numeric priority
        pprio = groups[k].get("priority")
        if pprio is not None:
            adj = pprio if parent_priority_dir == "asc" else -pprio
            return (1, adj, "", k)
        # parent rank or min child rank
        prank = groups[k].get("rank","") or groups[k].get("_min_child_rank","") or "~"
        if parent_dir == "desc":
            # invert rank by sorting on a tuple that will place lexicographically larger earlier
            return (2, "", "".join(chr(255 - ord(c)) for c in prank), k)  # cheap invert
        return (2, "", prank, k)

    ordered = sorted(groups.keys(), key=sort_group_key)

    if debug:
        print(f"Ordering with parent_rank_field={parent_rank_field or '(none)'} epic_rank_field={epic_rank_field or '(none)'} use_epic_rank={use_epic_rank} parent_dir={parent_dir} child_dir={child_dir} parent_priority_field={parent_priority_field or '(none)'} child_priority_field={child_priority_field or '(none)'}")
        if order_map:
            print("Explicit order map detected (lower index = higher priority):")
            print("  " + ", ".join([f"{k}:{v}" for k,v in list(order_map.items())[:30]]))
        print("Parent group order:")
        for k in ordered:
            meta = groups[k]
            print(f"  {k}: priority={meta.get('priority', '')} prank={meta.get('rank','')} source={meta.get('rank_source','')} min_child={meta.get('_min_child_rank','')} title={meta.get('title','')} size={len(meta['issues'])}")
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

        def child_key(it):
            cprio = it.get("_child_priority_cache")
            if cprio is not None:
                adj = cprio if child_priority_dir == "asc" else -cprio
                return (0, adj, it.get("key") or "")
            # rank fallback
            rk = it.get("_child_rank_cache") or "~"
            if child_dir == "desc":
                rk = "".join(chr(255 - ord(c)) for c in rk)
            return (1, rk, it.get("key") or "")

        children.sort(key=child_key)

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
    if len(text) <= max_len: return [text]
    chunks, cur, n = [], [], 0
    for line in text.splitlines(False):
        if n + len(line) + 1 > max_len:
            chunks.append("\n".join(cur)); cur, n = [line], len(line) + 1
        else:
            cur.append(line); n += len(line) + 1
    if cur: chunks.append("\n".join(cur))
    return chunks

def post_to_teams(webhook_url: str, title: str, markdown_text: str) -> None:
    payload = {"@type":"MessageCard","@context":"http://schema.org/extensions",
               "summary":title,"themeColor":"0076D7","title":title,"text":markdown_text}
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

    parent_rank_field = env_or_config("PARENT_RANK_FIELD") or ""
    epic_rank_field = env_or_config("PARENT_EPIC_RANK_FIELD") or ""
    issue_rank_field = env_or_config("ISSUE_RANK_FIELD") or ""
    auto_detect = parse_bool(env_or_config("AUTO_DETECT_RANK") or "true", default=True)
    auto_detect_epic = parse_bool(env_or_config("AUTO_DETECT_EPIC_RANK") or "true", default=True)
    use_epic_rank = parse_bool(env_or_config("USE_EPIC_RANK_FOR_EPICS") or "true", default=True)

    parent_dir = (env_or_config("PARENT_RANK_DIRECTION") or "desc").strip().lower()
    child_dir = (env_or_config("CHILD_RANK_DIRECTION") or "asc").strip().lower()

    # Numeric priority fields
    parent_priority_field = env_or_config("PARENT_PRIORITY_FIELD") or ""
    parent_priority_dir = (env_or_config("PARENT_PRIORITY_DIRECTION") or "asc").strip().lower()
    child_priority_field = env_or_config("CHILD_PRIORITY_FIELD") or ""
    child_priority_dir = (env_or_config("CHILD_PRIORITY_DIRECTION") or "asc").strip().lower()

    # Explicit order
    parent_order = env_or_config("PARENT_ORDER") or ""
    parent_order_mode = (env_or_config("PARENT_ORDER_MODE") or "auto").strip().lower()

    sort_children = parse_bool(env_or_config("SORT_CHILDREN_BY_RANK") or "true", default=True)
    debug = parse_bool(env_or_config("DEBUG_ORDER") or "false", default=False)

    missing = [n for n, v in [("JIRA_BASE_URL", base), ("JIRA_EMAIL", email), ("JIRA_API_TOKEN", token),
                               ("JIRA_JQL", jql), ("TEAMS_WEBHOOK_URL", webhook)] if not v]
    if missing:
        print("Missing required configuration values:", ", ".join(missing), file=sys.stderr); sys.exit(2)

    # Field auto-detect (Rank/Epic Rank) if not provided
    if auto_detect or auto_detect_epic:
        discovered = detect_fields(base, email, token, debug=debug)
        if auto_detect and not parent_rank_field and discovered.get("rank"):
            parent_rank_field = discovered["rank"]
        if auto_detect and not issue_rank_field and discovered.get("rank"):
            issue_rank_field = discovered["rank"]
        if auto_detect_epic and not epic_rank_field and discovered.get("epic_rank"):
            epic_rank_field = discovered["epic_rank"]

    req_fields = compute_request_fields(fields, group_by_parent, issue_rank_field if sort_children else "", child_priority_field)

    print(f"Querying Jira with JQL: {jql}")
    if use_legacy:
        issues = fetch_legacy(base, email, token, jql, req_fields, max_issues, batch_size, expand)
    else:
        issues = fetch_enhanced(base, email, token, jql, req_fields, max_issues, batch_size, expand, reconcile_ids)

    # Enrich ALL parents with summary + ranks + numeric priority
    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields) or group_by_parent:
        bulk_fill_parent_summaries(base, email, token, issues, parent_rank_field, epic_rank_field, parent_priority_field)

    print(f"Fetched {len(issues)} issues")

    if group_by_parent:
        markdown = build_grouped_markdown(
            issues, fields, tzname, datefmt,
            strip_parent_summary=strip_parent_summary,
            issue_rank_field=(issue_rank_field if sort_children else ""),
            parent_rank_field=parent_rank_field,
            epic_rank_field=epic_rank_field,
            use_epic_rank=use_epic_rank,
            parent_dir=parent_dir,
            child_dir=child_dir,
            parent_order=parent_order,
            parent_order_mode=parent_order_mode,
            parent_priority_field=parent_priority_field,
            child_priority_field=child_priority_field,
            parent_priority_dir=parent_priority_dir,
            child_priority_dir=child_priority_dir,
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
