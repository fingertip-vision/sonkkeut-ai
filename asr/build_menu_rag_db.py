"""Export a public menu response to the same scoped SQLite schema used on Android."""
import argparse
import json
import sqlite3
from pathlib import Path


def build(source: Path, destination: Path, server: str):
    menu = json.loads(source.read_text(encoding='utf-8-sig'))
    assert len(menu['items']) <= 1000
    documents = []
    names = set()
    for item in menu['items']:
        assert isinstance(item['name'], str) and 1 <= len(item['name']) <= 80 and item['name'] not in names
        assert isinstance(item['sold_out'], bool) and isinstance(item['aliases'], list)
        assert len(item['aliases']) <= 100 and all(isinstance(a, str) and 1 <= len(a) <= 80 for a in item['aliases'])
        names.add(item['name'])
        documents.append({k: item[k] for k in ('name', 'aliases', 'sold_out')})
        documents[-1]['category'] = item.get('category', '')
    scope = json.dumps([server.rstrip('/'), menu['store_code'], menu['menu_version']], ensure_ascii=False, separators=(',', ':'))
    payload = json.dumps(documents, ensure_ascii=False, separators=(',', ':'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as db:
        db.execute('CREATE TABLE IF NOT EXISTS catalog(scope TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO catalog(scope,payload) VALUES (?,?)', (scope, payload))
        db.execute('PRAGMA user_version = 1')
        assert json.loads(db.execute('SELECT payload FROM catalog WHERE scope=?', (scope,)).fetchone()[0]) == documents
    return {'database': destination.name, 'scope': scope, 'documents': len(documents), 'audio_or_transcripts_stored': False}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--server', required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.menu, args.out, args.server), ensure_ascii=False))
