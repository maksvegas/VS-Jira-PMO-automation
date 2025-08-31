#!/usr/bin/env python3
"""
jira_to_teams_report.py — Jira → Teams grouped report
v2.9 — adds TEAMS_MESSAGE_MODE="list" (numbered parents + bulleted children)

List mode formatting (Teams-friendly Markdown):
1. **PARENT_KEY — Parent Summary**
   - CHILD_KEY — Child Summary (Status: X, **Assignee**: Y, Updated: Z)

Env flags you may care about:
- TEAMS_MESSAGE_MODE: plain | adaptive | list
- TEAMS_SINGLE_MESSAGE: "true" (combine into one post when possible)
- CHILD_LIST_FIELDS: CSV of fields for children (default: key,summary,status,assignee,updated)
"""

import os, sys, json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import requests

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

VERSION = "jira_to_teams_report.py v2.9"

# ---------------- Helpers ----------------

def env(name: str, default=None):
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
        # Rank (LexoRank custom field typically)
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
        # Epic Rank, if present
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

# ------------- Parent enrichment -------------

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
                               parent_rank_field: str, epic_rank_field: str, parent_priority_field: str = "",
                               debug: bool = False, force_get: bool = False) -> None:
    want_keys = set()
    for iss in issues:
        fields = iss.get("fields") or {}
        parent = fields.get("parent")
        if isinstance(parent, dict) and parent.get("key"):
            want_keys.add(parent["key"])
    if not want_keys:
        return

    headers, auth = jira_auth(email, token)

    if force_get:
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
                p["_summary_cache"] = pf.get("summary", p.get("_summary_cache",""))
                if parent_priority_field: p["_priority_num_cache"] = pf.get(parent_priority_field, p.get("_priority_num_cache", None))
                if parent_rank_field:     p["_rank_cache"] = pf.get(parent_rank_field, p.get("_rank_cache", None))
                if epic_rank_field:       p["_epic_rank_cache"] = pf.get(epic_rank_field, p.get("_epic_rank_cache", None))
        return

    url = f"{base}/rest/api/3/search/jql"
    fetch_fields = ["summary"]
    if parent_rank_field: fetch_fields.append(parent_rank_field)
    if epic_rank_field and epic_rank_field not in fetch_fields: fetch_fields.append(epic_rank_field)
    if parent_priority_field and parent_priority_field not in fetch_fields: fetch_fields.append(parent_priority_field)
    payload = {
        "jql": "key in (" + ",".join(sorted(want_keys)) + ")",
        "fields": fetch_fields,
        "maxResults": len(want_keys),
    }

    by_key: Dict[str, Dict[str, Any]] = {}
    try:
        r = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if r.status_code == 200:
            data = r.json()
            for it in data.get("issues", []):
                k = it.get("key"); flds = it.get("fields") or {}
                meta = {"summary": flds.get("summary", "")}
                if parent_rank_field:     meta["rank"]      = flds.get(parent_rank_field)
                if epic_rank_field:       meta["epic_rank"] = flds.get(epic_rank_field)
                if parent_priority_field: meta["priority_num"] = flds.get(parent_priority_field)
                by_key[k] = meta
        else:
            print(f"Bulk parent /search failed: {r.status_code} {r.text[:160]}")

        for iss in issues:
            p = (iss.get("fields") or {}).get("parent")
            if isinstance(p, dict) and p.get("key"):
                meta = by_key.get(p["key"], {})
                p["_summary_cache"] = meta.get("summary", p.get("_summary_cache", ""))
                if "rank" in meta:        p["_rank_cache"]       = meta.get("rank", p.get("_rank_cache", None))
                if "epic_rank" in meta:   p["_epic_rank_cache"]  = meta.get("epic_rank", p.get("_epic_rank_cache", None))
                if "priority_num" in meta:p["_priority_num_cache"]= meta.get("priority_num", p.get("_priority_num_cache", None))

        missing = set()
        for iss in issues:
            p = (iss.get("fields") or {}).get("parent")
            if isinstance(p, dict) and p.get("key"):
                need_prio = parent_priority_field and p.get("_priority_num_cache") in (None, "", [])
                need_rank = parent_rank_field and p.get("_rank_cache") in (None, "", [])
                need_epic = epic_rank_field and p.get("_epic_rank_cache") in (None, "", [])
                if need_prio or need_rank or need_epic:
                    missing.add(p["key"])

        if missing:
            want_fields = ["summary"]
            if parent_priority_field: want_fields.append(parent_priority_field)
            if parent_rank_field:     want_fields.append(parent_rank_field)
            if epic_rank_field:       want_fields.append(epic_rank_field)
            per = fetch_parent_fields_individually(base, email, token, missing, want_fields, debug=debug)
            for iss in issues:
                p = (iss.get("fields") or {}).get("parent")
                if isinstance(p, dict) and p.get("key") and p["key"] in per:
                    pf = per[p["key"]]
                    p["_summary_cache"] = pf.get("summary", p.get("_summary_cache",""))
                    if parent_priority_field: p["_priority_num_cache"] = pf.get(parent_priority_field, p.get("_priority_num_cache", None))
                    if parent_rank_field:     p["_rank_cache"] = pf.get(parent_rank_field, p.get("_rank_cache", None))
                    if epic_rank_field:       p["_epic_rank_cache"] = pf.get(epic_rank_field, p.get("_epic_rank_cache", None))

    except Exception as e:
        print(f"bulk_fill_parent_summaries exception: {e}")

# ------------- Rendering + Ordering -------------

def format_val(issue: Dict[str, Any], f: str, tz: str, df: str) -> str:
    v = extract_field(issue, f, tz, df)
    return v or ""

def build_ordered_groups(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str,
                         strip_parent_summary: bool, issue_rank_field: str,
                         parent_rank_field: str, epic_rank_field: str, use_epic_rank: bool,
                         parent_dir: str, child_dir: str,
                         parent_order: str, parent_order_mode: str,
                         parent_priority_field: str, child_priority_field: str,
                         parent_priority_dir: str, child_priority_dir: str,
                         parent_priority_agg: str, debug=False):
    # Reuse grouping & sort logic
    groups = group_issues_by_parent(issues, issue_rank_field, parent_rank_field, epic_rank_field, use_epic_rank,
                                    parent_priority_field=parent_priority_field, child_priority_field=child_priority_field)

    # derive parent priority if requested
    if parent_priority_agg in {"min","avg","max"}:
        for k, meta in groups.items():
            if meta.get("priority") is None:
                vals = [i.get("_child_priority_cache") for i in meta["issues"] if i.get("_child_priority_cache") is not None]
                if vals:
                    if parent_priority_agg == "min": meta["priority"] = min(vals)
                    elif parent_priority_agg == "max": meta["priority"] = max(vals)
                    else: meta["priority"] = sum(vals)/len(vals)

    order_map = {}
    if parent_order:
        raw = [x.strip() for x in parent_order.split(",") if x.strip()]
        for idx, item in enumerate(raw):
            order_map[item.lower()] = idx

    def explicit_index(k: str):
        meta = groups[k]
        title = (meta.get("title") or "").lower()
        if parent_order_mode in ("auto","keys"):
            if k.lower() in order_map: return order_map[k.lower()]
        if parent_order_mode in ("auto","titles"):
            if title and title.lower() in order_map: return order_map[title.lower()]
        return None

    def sort_group_key(k: str):
        if k == "NO_PARENT": return (9999, 0, "", k)
        idx = explicit_index(k)
        if idx is not None:
            return (0, idx, "", k)
        pprio = groups[k].get("priority")
        if pprio is not None:
            adj = pprio if parent_priority_dir == "asc" else -pprio
            return (1, adj, "", k)
        prank = groups[k].get("rank","") or groups[k].get("_min_child_rank","") or "~"
        if parent_dir == "desc":
            return (2, "", "".join(chr(255 - ord(c)) for c in prank), k)
        return (2, "", prank, k)

    ordered_keys = sorted(groups.keys(), key=sort_group_key)

    # child ordering
    def child_key_it(it):
        cprio = it.get("_child_priority_cache")
        if cprio is not None:
            adj = cprio if child_priority_dir == "asc" else -cprio
            return (0, adj, it.get("key") or "")
        rk = it.get("_child_rank_cache") or "~"
        if child_dir == "desc":
            rk = "".join(chr(255 - ord(c)) for c in rk)
        return (1, rk, it.get("key") or "")

    for k in ordered_keys:
        groups[k]["issues"].sort(key=child_key_it)

    return ordered_keys, groups

def build_list_markdown(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str,
                        strip_parent_summary: bool, issue_rank_field: str,
                        parent_rank_field: str, epic_rank_field: str, use_epic_rank: bool,
                        parent_dir: str, child_dir: str,
                        parent_order: str, parent_order_mode: str,
                        parent_priority_field: str, child_priority_field: str,
                        parent_priority_dir: str, child_priority_dir: str,
                        parent_priority_agg: str,
                        child_list_fields: List[str],
                        debug=False) -> str:
    ordered_keys, groups = build_ordered_groups(
        issues, fields, tzname, datefmt, strip_parent_summary,
        issue_rank_field, parent_rank_field, epic_rank_field, use_epic_rank,
        parent_dir, child_dir,
        parent_order, parent_order_mode,
        parent_priority_field, child_priority_field,
        parent_priority_dir, child_priority_dir,
        parent_priority_agg, debug=debug
    )

    # Build numbered + bullets
    lines = []
    n = 1
    for k in ordered_keys:
        meta = groups[k]
        if k == "NO_PARENT":
            parent_title = "**No Parent**"
        else:
            parent_summary = meta.get("title") or ""
            parent_title = f"**{k} — {parent_summary}**" if parent_summary else f"**{k}**"
        lines.append(f"{n}. {parent_title}")
        n += 1
        # children bullets
        for it in meta["issues"]:
            # assemble child display by requested fields
            parts = []
            for fld in child_list_fields:
                name = fld.strip().lower()
                if name == "key":
                    parts.append(it.get("key",""))
                elif name == "summary":
                    parts.append(format_val(it, "summary", tzname, datefmt))
                elif name == "status":
                    val = format_val(it, "status", tzname, datefmt)
                    if val: parts.append(f"Status: {val}")
                elif name == "assignee":
                    who = format_val(it, "assignee", tzname, datefmt)
                    if who: parts.append(f"Assignee: **{who}**")
                elif name == "updated":
                    val = format_val(it, "updated", tzname, datefmt)
                    if val: parts.append(f"Updated: {val}")
                elif name in {"parentsummary","parent_summary"}:
                    # skip in child line
                    continue
                else:
                    val = format_val(it, name, tzname, datefmt)
                    if val: parts.append(f"{fld}: {val}")
            txt = " — ".join([p for p in parts if p])
            if not txt:
                txt = it.get("key","")
            lines.append(f"   - {txt}")
        # blank line after each parent block
        lines.append("")
    return "\n".join(lines).rstrip()

# ---- ASCII / Adaptive helpers from prior version ----

def to_monospace_table(headers: List[str], rows: List[List[str]], max_col_width=60) -> str:
    if not headers: return "```\n(no columns)\n```"
    widths = []
    for i, h in enumerate(headers):
        longest = len(h)
        for r in rows:
            if i < len(r):
                v = "" if r[i] is None else str(r[i]).replace("\n", " ")
                if len(v) > longest: longest = len(v)
        widths.append(min(longest, max_col_width))
    def cut(s, w):
        s = "" if s is None else str(s).replace("\n", " ")
        return s[:w]
    def fmt_row(cells):
        return " | ".join(cut(c, w).ljust(w) for c, w in zip(cells, widths))
    lines = []
    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt_row([r[i] if i < len(r) else "" for i in range(len(headers))]))
    return "```\n" + "\n".join(lines) + "\n```"

def post_to_teams_messagecard(webhook_url: str, title: str, text_block: str) -> None:
    payload = {"@type":"MessageCard","@context":"http://schema.org/extensions",
               "summary":title,"themeColor":"0076D7","title":title,"text":text_block}
    resp = requests.post(webhook_url, json=payload, timeout=45)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Teams webhook error {resp.status_code}: {resp.text}")

def post_to_teams_adaptive_grid(webhook_url: str, title: str, sections: List[Tuple[str, List[str], List[List[str]]]]) -> None:
    body = [{"type":"Container","items":[{"type":"TextBlock","text": title,"weight":"Bolder","size":"Medium"}]}]
    for heading, cols, rows in sections:
        body.append({"type":"Container","items":[{"type":"TextBlock","text": f"**{heading}**","wrap": True}]})
        header_cols = [{"type":"TextBlock","text": f"**{c}**","wrap": True} for c in cols]
        body.append({"type":"Container","items":[{"type":"ColumnSet","columns":[{"type":"Column","width":"stretch","items":[hc]} for hc in header_cols]}]})
        for r in rows:
            cells = [{"type":"TextBlock","text": ("" if c is None else str(c)), "wrap": True} for c in r]
            body.append({"type":"Container","items":[{"type":"ColumnSet","columns":[{"type":"Column","width":"stretch","items":[cell]} for cell in cells]}]})

    payload = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "contentUrl": None,
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": body
            }
        }]
    }
    resp = requests.post(webhook_url, json=payload, timeout=45)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Teams webhook error {resp.status_code}: {resp.text}")

# ---------------- Main ----------------

def main():
    base = env("JIRA_BASE_URL")
    email = env("JIRA_EMAIL")
    token = env("JIRA_API_TOKEN")
    jql = env("JIRA_JQL")
    fields_csv = env("JIRA_FIELDS")
    fields = [s.strip() for s in fields_csv.split(",")] if fields_csv else ["key","summary","parentSummary","status","assignee","updated"]
    webhook = env("TEAMS_WEBHOOK_URL")
    title = env("TITLE", "Jira Report")
    datefmt = env("DATE_FORMAT", "%Y-%m-%d %H:%M")
    tzname = env("TIMEZONE", "UTC")
    max_issues = int(env("MAX_ISSUES", "500"))
    batch_size = int(env("BATCH_SIZE", "100"))
    use_legacy = parse_bool(env("JIRA_USE_LEGACY"), default=False)
    expand = env("EXPAND", "")
    reconcile_ids = parse_reconcile_ids(env("RECONCILE_ISSUE_IDS", ""))
    group_by_parent = parse_bool(env("GROUP_BY_PARENT", "true"), default=True)
    strip_parent_summary = parse_bool(env("GROUP_STRIP_PARENT_SUMMARY", "true"), default=True)

    parent_rank_field = env("PARENT_RANK_FIELD", "")
    epic_rank_field = env("PARENT_EPIC_RANK_FIELD", "")
    issue_rank_field = env("ISSUE_RANK_FIELD", "")
    auto_detect = parse_bool(env("AUTO_DETECT_RANK", "true"), default=True)
    auto_detect_epic = parse_bool(env("AUTO_DETECT_EPIC_RANK", "true"), default=True)
    use_epic_rank = parse_bool(env("USE_EPIC_RANK_FOR_EPICS", "true"), default=True)

    parent_dir = env("PARENT_RANK_DIRECTION", "desc").strip().lower()
    child_dir = env("CHILD_RANK_DIRECTION", "asc").strip().lower()

    parent_priority_field = env("PARENT_PRIORITY_FIELD", "")
    parent_priority_dir = env("PARENT_PRIORITY_DIRECTION", "asc").strip().lower()
    child_priority_field = env("CHILD_PRIORITY_FIELD", "")
    child_priority_dir = env("CHILD_PRIORITY_DIRECTION", "asc").strip().lower()
    parent_priority_agg = env("PARENT_PRIORITY_AGG", "").strip().lower()

    parent_order = env("PARENT_ORDER", "")
    parent_order_mode = env("PARENT_ORDER_MODE", "auto").strip().lower()

    sort_children = parse_bool(env("SORT_CHILDREN_BY_RANK", "true"), default=True)
    debug = parse_bool(env("DEBUG_ORDER", "false"), default=False)
    debug_parent = parse_bool(env("DEBUG_PARENT_PRIORITY", "false"), default=False)
    force_parent_get = parse_bool(env("PARENT_ENRICH_FORCE_GET", "false"), default=False)

    teams_mode = env("TEAMS_MESSAGE_MODE", "list").strip().lower()  # default to list here if not provided
    chunk_limit = int(env("TEAMS_CHUNK_LIMIT", "20000"))
    single_msg = parse_bool(env("TEAMS_SINGLE_MESSAGE", "true"), default=True)
    rows_per_card = int(env("TEAMS_ADAPTIVE_ROWS_PER_CARD", "60"))

    child_list_fields_csv = env("CHILD_LIST_FIELDS", "key,summary,status,assignee,updated")
    child_list_fields = [s.strip() for s in child_list_fields_csv.split(",") if s.strip()]

    missing = [n for n, v in [("JIRA_BASE_URL", base), ("JIRA_EMAIL", email), ("JIRA_API_TOKEN", token),
                               ("JIRA_JQL", jql), ("TEAMS_WEBHOOK_URL", webhook)] if not v]
    if missing:
        print("Missing required configuration values:", ", ".join(missing), file=sys.stderr); sys.exit(2)

    if auto_detect or auto_detect_epic:
        discovered = detect_fields(base, email, token, debug=debug)
        if auto_detect and not parent_rank_field and discovered.get("rank"):
            parent_rank_field = discovered["rank"]
        if auto_detect and not issue_rank_field and discovered.get("rank"):
            issue_rank_field = discovered["rank"]
        if auto_detect_epic and not epic_rank_field and discovered.get("epic_rank"):
            epic_rank_field = discovered["epic_rank"]

    req_fields = compute_request_fields(fields, group_by_parent, issue_rank_field if sort_children else "", child_priority_field)

    print(VERSION)
    print(f"Querying Jira with JQL: {jql}")
    if use_legacy:
        issues = fetch_legacy(base, email, token, jql, req_fields, max_issues, batch_size, expand)
    else:
        issues = fetch_enhanced(base, email, token, jql, req_fields, max_issues, batch_size, expand, reconcile_ids)

    # Enrich ALL parents
    if any(f.lower() in {"parentsummary", "parent_summary"} for f in fields) or group_by_parent:
        parent_keys = sorted({
            (iss.get("fields") or {}).get("parent", {}).get("key")
            for iss in issues
            if isinstance((iss.get("fields") or {}).get("parent"), dict)
            and (iss.get("fields") or {}).get("parent").get("key")
        })
        print(f"Enrichment flags -> force_get={force_parent_get} debug_parent={debug_parent} parents_detected={len(parent_keys)}")
        if parent_keys[:5]:
            print(f"Sample parents: {parent_keys[:5]}")
        bulk_fill_parent_summaries(
            base, email, token, issues,
            parent_rank_field, epic_rank_field, parent_priority_field,
            debug=debug_parent, force_get=force_parent_get
        )

    print(f"Fetched {len(issues)} issues")

    # Render according to mode
    if teams_mode == "list":
        text = build_list_markdown(
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
            parent_priority_agg=parent_priority_agg,
            child_list_fields=child_list_fields,
            debug=debug
        )
        # Chunk if needed
        if len(text) <= chunk_limit:
            post_to_teams_messagecard(webhook, title, text)
            print("Posted single list message")
        else:
            # chunk safely at parent boundaries
            parts = text.split("\n\n")
            cur, acc, part = [], 0, 1
            for block in parts:
                block_len = len(block) + 2  # + two newlines
                if acc + block_len > chunk_limit and cur:
                    payload = "\n\n".join(cur)
                    post_to_teams_messagecard(webhook, f"{title} (part {part})", payload)
                    print(f"Posted list part {part}")
                    cur, acc, part = [block], block_len, part+1
                else:
                    cur.append(block); acc += block_len
            if cur:
                payload = "\n\n".join(cur)
                post_to_teams_messagecard(webhook, f"{title} (part {part})", payload)
                print(f"Posted list part {part}")
    else:
        # Fall back to prior modes for completeness
        # Build tabular sections (kept from earlier versions)
        def build_rows_flat(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str) -> Tuple[List[str], List[List[str]]]:
            cols = list(fields)
            rows = [[extract_field(issue, f, tzname, datefmt) for f in cols] for issue in issues]
            return cols, rows

        def build_rows_grouped(issues: List[Dict[str, Any]], fields: List[str], tzname: str, datefmt: str,
                               strip_parent_summary: bool, issue_rank_field: str,
                               parent_rank_field: str, epic_rank_field: str, use_epic_rank: bool,
                               parent_dir: str, child_dir: str,
                               parent_order: str, parent_order_mode: str,
                               parent_priority_field: str, child_priority_field: str,
                               parent_priority_dir: str, child_priority_dir: str,
                               parent_priority_agg: str, debug=False) -> List[Tuple[str, List[str], List[List[str]]]]:
            ok, groups = build_ordered_groups(
                issues, fields, tzname, datefmt, strip_parent_summary, issue_rank_field, parent_rank_field, epic_rank_field,
                use_epic_rank, parent_dir, child_dir, parent_order, parent_order_mode,
                parent_priority_field, child_priority_field, parent_priority_dir, child_priority_dir, parent_priority_agg, debug=debug
            )
            fields_local = list(fields)
            if strip_parent_summary:
                fields_local = [f for f in fields_local if f.lower() not in {"parentsummary","parent_summary"}]
            sections = []
            for pkey in ok:
                meta = groups[pkey]
                title = meta.get("title") or ""
                heading = "No Parent" if pkey == "NO_PARENT" else (f"{pkey} — {title}" if title else pkey)
                children = list(meta["issues"])
                cols = list(fields_local)
                rows = [[extract_field(issue, f, tzname, datefmt) for f in cols] for issue in children]
                sections.append((heading, cols, rows))
            return sections

        # Section build
        if group_by_parent:
            sections = build_rows_grouped(
                issues, fields, tzname, datefmt,
                strip_parent_summary=strip_parent_summary,
                issue_rank_field=(issue_rank_field if parse_bool(env("SORT_CHILDREN_BY_RANK","true"), True) else ""),
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
                parent_priority_agg=parent_priority_agg,
                debug=debug
            )
        else:
            cols, rows = build_rows_flat(issues, fields, tzname, datefmt)
            sections = [("All Issues", cols, rows)]

        # Respect TEAMS_MESSAGE_MODE for plain/adaptive
        mode = teams_mode
        if mode == "adaptive":
            post_to_teams_adaptive_grid(webhook, title, sections)
            print("Posted adaptive grid")
        else:
            blocks = []
            for heading, cols, rows in sections:
                blocks.append(f"**{heading}**\n\n" + to_monospace_table(cols, rows))
            full_text = "\n\n".join(blocks)
            post_to_teams_messagecard(webhook, title, full_text)
            print("Posted plain ASCII tables")

# Reuse earlier extract_field and group_issues_by_parent from v2.8
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
            pprio_val = None
            if parent_priority_field:
                pprio_val = pf.get(parent_priority_field)
                if pprio_val is None:
                    pprio_val = parent.get("_priority_num_cache", None)
                try:
                    pprio_val = float(pprio_val) if pprio_val is not None and pprio_val != "" else None
                except Exception:
                    pprio_val = None
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
    for k, meta in groups.items():
        cr = [i.get("_child_rank_cache") for i in meta["issues"] if i.get("_child_rank_cache")]
        meta["_min_child_rank"] = min(cr) if cr else ""
    return groups

if __name__ == "__main__":
    main()
