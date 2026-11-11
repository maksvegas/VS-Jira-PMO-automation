"""Load an ALABS weekly CSV snapshot into BigQuery, idempotently per run/source/key."""
import csv
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from google.cloud import bigquery

DATE_COLS = {'Snapshot Date', 'Due Date', 'Target start', 'Target end'}
TIMESTAMP_COLS = {'Created', 'Updated', 'Status Category Changed'}
# Preserve source CSV header names as BigQuery-friendly snake_case.
def field_name(header):
    return re.sub(r'[^a-z0-9]+', '_', header.lower()).strip('_')

def parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f'Invalid date {value!r}') from exc

def parse_timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError as exc:
        raise ValueError(f'Invalid timestamp {value!r}') from exc

def main(csv_path):
    project = os.environ['GCP_PROJECT_ID']
    dataset_id = os.environ.get('BQ_DATASET', 'jira_reporting')
    table_id = os.environ.get('BQ_TABLE', 'portfolio_snapshots')
    location = os.environ.get('BQ_LOCATION', 'europe-west2')
    client = bigquery.Client(project=project, location=location)
    dataset_ref = f'{project}.{dataset_id}'
    target = f'{dataset_ref}.{table_id}'
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError('CSV missing header row')
        mapped = {h: field_name(h) for h in headers}
        if len(set(mapped.values())) != len(mapped):
            raise ValueError('CSV column names collide after normalisation')
        required = {'Snapshot Date', 'Snapshot Run ID', 'Source', 'Portfolio', 'Key'}
        if not required.issubset(headers):
            raise ValueError(f'Missing required columns: {required - set(headers)}')
        schema = [bigquery.SchemaField(mapped[h], 'DATE' if h in DATE_COLS else 'TIMESTAMP' if h in TIMESTAMP_COLS else 'STRING') for h in headers]
        records = []
        for row in reader:
            out = {}
            for header in headers:
                value = (row[header] or '').strip()
                out[mapped[header]] = parse_date(value) if header in DATE_COLS else parse_timestamp(value) if header in TIMESTAMP_COLS else (value or None)
            records.append(out)
    if not records:
        raise ValueError('No issues returned. Refusing to load an empty snapshot.')
    keys = [(r['snapshot_run_id'], r['source'], r['portfolio'], r['key']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate (run, source, portfolio, issue key) values in snapshot')

    # Dataset is created explicitly in setup; permissions only require dataset-level editor.
    client.get_dataset(dataset_ref)
    try:
        table = client.get_table(target)
        existing = {f.name: f.field_type for f in table.schema}
        expected = {f.name: f.field_type for f in schema}
        if existing != expected:
            raise ValueError(f'Existing table schema differs. Existing={existing}; Expected={expected}')
    except Exception as exc:
        from google.api_core.exceptions import NotFound
        if not isinstance(exc, NotFound):
            raise
        table = bigquery.Table(target, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(field='snapshot_date')
        table.clustering_fields = ['source', 'portfolio', 'key']
        client.create_table(table)

    stage = f'{dataset_ref}._portfolio_stage_{uuid4().hex}'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.jsonl') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()
        with open(f.name, 'rb') as binary:
            job = client.load_table_from_file(binary, stage, job_config=bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema, write_disposition='WRITE_TRUNCATE'), location=location)
            job.result()
    try:
        # Rerunning the same GitHub run replaces matching records; distinct runs retain history.
        columns = [f.name for f in schema]
        update = ', '.join(f'T.{c} = S.{c}' for c in columns if c not in ('snapshot_run_id', 'source', 'portfolio', 'key'))
        insert_cols = ', '.join(columns)
        insert_vals = ', '.join(f'S.{c}' for c in columns)
        sql = f'''MERGE `{target}` T USING `{stage}` S
        ON T.snapshot_run_id = S.snapshot_run_id AND T.source = S.source
           AND T.portfolio = S.portfolio AND T.key = S.key
        WHEN MATCHED THEN UPDATE SET {update}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})'''
        client.query(sql, location=location).result()
        print(f'Loaded {len(records)} Jira issues into {target}')
    finally:
        client.delete_table(stage, not_found_ok=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/upload_snapshot_to_bigquery.py <csv_path>')
    main(Path(sys.argv[1]))
