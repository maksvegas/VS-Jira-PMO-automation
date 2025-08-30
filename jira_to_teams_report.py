#!/usr/bin/env python3
"""
jira_to_teams_report.py

Autogenerate a Jira report (via JQL) with specific fields and post it
to a Microsoft Teams channel as a text message using an Incoming Webhook.

Now updated to use Atlassian's ENHANCED JQL SEARCH endpoint:
  - /rest/api/3/search/jql  (GET/POST)
with optional read-after-write consistency via reconcileIssues.

Usage:
  - Set the required environment variables (see below), or edit the CONFIG block.
  - Run: python jira_to_teams_report.py

Environment variables (override CONFIG if present):
  JIRA_BASE_URL          e.g. https://your-domain.atlassian.net
  JIRA_EMAIL             Atlassian account email (for API token auth)
  JIRA_API_TOKEN         API token from https://id.atlassian.com/manage-profile/security/api-tokens
  JIRA_JQL               A valid JQL string (quotes recommended)
  JIRA_FIELDS            Comma-separated Jira field IDs/names (e.g. key,summary,status,assignee,updated)
  TEAMS_WEBHOOK_URL      Teams Incoming Webhook URL
  MAX_ISSUES             Max issues to fetch (default 500)
  BATCH_SIZE             Page size for Jira search (default 100)
  TITLE                  Title shown at the top of the Teams message
  DATE_FORMAT            Python strftime format for date fields (default %Y-%m-%d %H:%M)
  TIMEZONE               IANA tz name for dates (default UTC)

  # Enhanced search options
  RECONCILE_ISSUES       Set "true" for stronger read-after-write consistency (default false)
  EXPAND                 Comma-separated expand params (e.g. render,transitions,changelog) (optional)
  JIRA_USE_LEGACY        If "true", uses deprecated /rest/api/3/search (not recommended)

Notes:
  - Teams webhooks support simple Markdown. This script posts a Markdown table.
  - Jira's REST API uses field IDs. For custom fields, use customfield_xxxxx.
  - The "key" field is always available from issue["key"] and not in issue["fields"].
  - The "status" name is at issue["fields"]["status"]["name"].
  - The "assignee" display name is at issue["fields"]["assignee"]["displayName"].
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# -------------------- CONFIG (can be overridden by environment variables) --------------------

CONFIG = {
    "JIRA_BASE_URL": "https://your-domain.atlassian.net",
    "JIRA_EMAIL": "you@example.com",
    "JIRA_API_TOKEN": "YOUR_API_TOKEN",
    "JIRA_JQL": "project = ABC AND updated >= -1d ORDER BY updated DESC",
    "JIRA_FIELDS": ["key", "summary", "status", "assignee", "updated"],
    "TEAMS_WEBHOOK_URL": "https://outlook.office.com/webhook/your-webhook-url",
    "MAX_ISSUES": 500,
    "BATCH_SIZE": 100,
    "TITLE": "Jira Report",
    "DATE_FORMAT": "%Y-%m-%d %H:%M",
    "TIMEZONE": "UTC",
    # Enhanced search options
    "RECONCILE_ISSUES": "false",
    "EXPAND": "",
    "JIRA_USE_LEGACY": "false",
}

# -------------- Helpers --------------

def env_or_config(name: str, default=None):
    return os.getenv(name, CONFIG.get(name, default))

def normalize_fields(fields: List[str]) -> List[str]:
    return [f for f in fields if f.lower() != "key"]

def parse_bool(val: str, default=False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def format_date(dt_str: str, tzname: str, fmt: str) -> str:
    if not dt_str:
        return ""
    try:
        dt_str = dt_str.replace("+0000", "+00:00")
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(dt_str)
        if tzname and ZoneInfo:
            dt = dt.astimezone(ZoneInfo(tzname))
        return dt.strftime(fmt)
    except Exception:
        return dt_str

def jira_auth_headers(email: str, api_token: str) -> (Dict[str, str], tuple):
    return {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }, (email, api_token)

def fetch_jira_issues(base_url: str, email: str, api_token: str, jql: str, fields: List[str],
                      max_issues: int, batch_size: int, reconcile: bool = False,
                      expand: str = "", use_legacy: bool = False) -> List[Dict[str, Any]]:
    normalized_fields = normalize_fields(fields)
    issues = []
    start_at = 0
    search_path = "/rest/api/3/search" if use_legacy else "/rest/api/3/search/jql"
    url = f"{base_url}{search_path}"
    headers, auth = jira_auth_headers(email, api_token)

    while True:
        limit = min(batch_size, max_issues - start_at)
        if limit <= 0:
            break

        payload = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": limit,
            "fields": normalized_fields
        }
        if not use_legacy and reconcile:
            payload["reconcileIssues"] = True
        if expand:
            payload["expand"] = [e.strip() for e in expand.split(",") if e.strip()]

        resp = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Jira API error {resp.status_code}: {resp.text}")
        data = resp.json()
        batch = data.get("issues", [])
        issues.extend(batch)
        start_at += len(batch)
        if start_at >= data.get("total", 0) or len(batch) == 0 or start_at >= max_issues:
            break
    return issues

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

def chunk_text(text: str, max_len: int = 25000) -> List[str]:
    if len(text) <= max_len:
        return [text]
    chunks, current, current_len = [], [], 0
    for line in text.splitlines(keepends=False):
        if current_len + len(line) + 1 > max_len:
            chunks.append("\n".join(current))
            current, current_len = [line], len(line) + 1
        else:
            current.append(line)
            current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
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
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Teams webhook error {resp.status_code}: {resp.text}")

def main():
    base_url = env_or_config("JIRA_BASE_URL")
    email = env_or_config("JIRA_EMAIL")
    api_token = env_or_config("JIRA_API_TOKEN")
    jql = env_or_config("JIRA_JQL")
    fields_csv = os.getenv("JIRA_FIELDS")
    fields = [s.strip() for s in fields_csv.split(",")] if fields_csv else CONFIG["JIRA_FIELDS"]
    webhook = env_or_config("TEAMS_WEBHOOK_URL")
    max_issues = int(env_or_config("MAX_ISSUES") or CONFIG["MAX_ISSUES"])
    batch_size = int(env_or_config("BATCH_SIZE") or CONFIG["BATCH_SIZE"])
    reconcile = parse_bool(env_or_config("RECONCILE_ISSUES"), default=False)
    expand = env_or_config("EXPAND") or ""
    use_legacy = parse_bool(env_or_config("JIRA_USE_LEGACY"), default=False)
    title = env_or_config("TITLE") or CONFIG["TITLE"]
    datefmt = env_or_config("DATE_FORMAT") or CONFIG["DATE_FORMAT"]
    tzname = env_or_config("TIMEZONE") or CONFIG["TIMEZONE"]

    missing = [n for n, v in [("JIRA_BASE_URL", base_url), ("JIRA_EMAIL", email), ("JIRA_API_TOKEN", api_token),
                               ("JIRA_JQL", jql), ("TEAMS_WEBHOOK_URL", webhook)] if not v]
    if missing:
        print("Missing required configuration values:", ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    print(f"Querying Jira (enhanced search) with JQL: {jql}")
    issues = fetch_jira_issues(base_url, email, api_token, jql, fields, max_issues, batch_size,
                               reconcile=reconcile, expand=expand, use_legacy=use_legacy)
    print(f"Fetched {len(issues)} issues")

    markdown_table = build_markdown_table(issues, fields, tzname, datefmt)

    chunks = chunk_text(markdown_table, max_len=25000)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_title = title if len(chunks) == 1 else f"{title} (part {idx}/{len(chunks)})"
        post_to_teams(webhook, chunk_title, chunk)
        print(f"Posted chunk {idx}/{len(chunks)} to Teams")

if __name__ == "__main__":
    main()
