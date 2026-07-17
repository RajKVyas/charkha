#!/usr/bin/env python3
"""CHARKHA news ingest: RSS/Atom -> cleaned JSONL + dataprep manifest.

This is the freshness firehose. It deliberately does not write token shards directly; it writes raw
JSONL docs that still pass through src/dataprep.py for normalization, quality filtering, PII scrub,
dedup, decontamination, and tokenizer-specific sharding.

Usage:
  python scripts/news_ingest.py --feeds configs/news_feeds.txt --out data/news/news.jsonl \
    --manifest configs/_src_manifests/src_local_news.yaml --limit 1000
  python scripts/news_ingest.py --selftest
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataprep import normalize, quality_filter, pii_scrub  # noqa: E402

UA = "CHARKHA-news-ingest/0.1 (+personal research crawler; contact: local)"
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
SPACE_RE = re.compile(r"\s+")


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node, names):
    names = {n.lower() for n in names}
    for ch in list(node):
        if _strip_ns(ch.tag) in names:
            return "".join(ch.itertext()).strip()
    return ""


def _parse_date(s: str) -> float | None:
    if not s:
        return None
    try:
        return email.utils.parsedate_to_datetime(s).timestamp()
    except Exception:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def rss_items(xml_text: str, feed_url: str = ""):
    root = ET.fromstring(xml_text)
    items = []
    for node in root.iter():
        tag = _strip_ns(node.tag)
        if tag not in ("item", "entry"):
            continue
        title = _child_text(node, ("title",))
        link = _child_text(node, ("link",))
        if not link:
            for ch in list(node):
                if _strip_ns(ch.tag) == "link":
                    link = ch.attrib.get("href", "")
                    if link:
                        break
        link = urllib.parse.urljoin(feed_url, link)
        summary = _child_text(node, ("description", "summary", "content", "encoded"))
        published = _child_text(node, ("pubDate", "published", "updated"))
        items.append(
            {
                "title": title,
                "url": link,
                "summary": summary,
                "published": published,
                "published_ts": _parse_date(published),
            }
        )
    return items


def html_to_text(raw: str) -> str:
    raw = SCRIPT_RE.sub(" ", raw)
    raw = re.sub(r"</(p|div|h\d|li|br|blockquote)>", "\n", raw, flags=re.I)
    raw = TAG_RE.sub(" ", raw)
    raw = html.unescape(raw)
    return normalize(SPACE_RE.sub(" ", raw))


def fetch_url(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "text/html,application/rss+xml"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(4_000_000)
        enc = r.headers.get_content_charset() or "utf-8"
    return data.decode(enc, errors="replace")


def doc_from_item(item: dict, fetch_pages: bool = True, timeout: int = 20) -> dict | None:
    text = html_to_text(item.get("summary") or "")
    if fetch_pages and item.get("url"):
        try:
            page_text = html_to_text(fetch_url(item["url"], timeout=timeout))
            if len(page_text.split()) > len(text.split()):
                text = page_text
        except Exception:
            pass
    title = normalize(item.get("title") or "")
    if title and title not in text[:500]:
        text = f"{title}\n\n{text}" if text else title
    text, _n_pii = pii_scrub(text)
    ok, reason = quality_filter(text, min_words=25, target_langs=["en"])
    if not ok:
        return None
    return {
        "id": "local/news-firehose",
        "text": text,
        "url": item.get("url", ""),
        "title": title,
        "published": item.get("published", ""),
        "license": "unknown",
        "source": urllib.parse.urlparse(item.get("url", "")).netloc,
        "quality": reason,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def load_feeds(path: str) -> list[str]:
    feeds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                feeds.append(line)
    return feeds


def write_manifest(path: str, jsonl_path: str, weight: float = 0.05):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data_root = os.path.join(ROOT, "data")
    if os.path.isabs(jsonl_path):
        rel = os.path.relpath(jsonl_path, data_root)
    else:
        rel = jsonl_path
        prefix = "data" + os.sep
        if rel == "data":
            rel = "."
        elif rel.startswith(prefix):
            rel = rel[len(prefix) :]
    text = (
        "sources:\n"
        "  - id: local/news-firehose\n"
        f"    data_files: {rel}\n"
        "    split: train\n"
        "    license: unknown\n"
        "    content_type: prose\n"
        "    role: freshness\n"
        f"    weight: {weight}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def ingest(
    feeds_path: str,
    out_path: str,
    limit: int = 1000,
    since_hours: float | None = None,
    fetch_pages: bool = True,
    manifest_path: str | None = None,
):
    feeds = load_feeds(feeds_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cutoff = time.time() - since_hours * 3600 if since_hours else None
    seen = set()
    kept = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for feed in feeds:
            if kept >= limit:
                break
            try:
                xml_text = fetch_url(feed)
                items = rss_items(xml_text, feed_url=feed)
            except Exception as e:
                print(f"[news] skip feed {feed}: {e}")
                continue
            for item in items:
                if kept >= limit:
                    break
                if cutoff and item.get("published_ts") and item["published_ts"] < cutoff:
                    continue
                key = item.get("url") or (item.get("title", "") + item.get("published", ""))
                h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                doc = doc_from_item(item, fetch_pages=fetch_pages)
                if not doc:
                    continue
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
                if kept % 100 == 0:
                    print(f"[news] kept {kept} docs")
    if manifest_path:
        write_manifest(manifest_path, out_path)
    print(f"[news] wrote {kept} docs -> {out_path}")
    return kept


def selftest():
    print("CHARKHA news ingest self-test")
    sample = """<?xml version='1.0'?><rss><channel><item>
      <title>Small Model Learns Quickly</title>
      <link>https://example.com/a</link>
      <pubDate>Sat, 04 Jul 2026 12:00:00 GMT</pubDate>
      <description><![CDATA[<p>A small language model learned from clean stories and then improved through careful distillation. The report explained the method in simple language for readers. It included enough ordinary English words to pass the quality gate.</p>]]></description>
    </item></channel></rss>"""
    items = rss_items(sample, "https://example.com/feed.xml")
    doc = doc_from_item(items[0], fetch_pages=False)
    tmp = os.path.join("/tmp", "charkha_news_selftest")
    os.makedirs(tmp, exist_ok=True)
    manifest = os.path.join(tmp, "src_local_news.yaml")
    out = os.path.join(tmp, "news.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc) + "\n")
    write_manifest(manifest, out)
    rel_manifest = os.path.join(tmp, "src_local_news_rel.yaml")
    write_manifest(rel_manifest, "data/news/news.jsonl")
    manifest_text = open(manifest, encoding="utf-8").read()
    rel_manifest_text = open(rel_manifest, encoding="utf-8").read()
    checks = {
        "rss item parsed": len(items) == 1 and items[0]["url"] == "https://example.com/a",
        "html cleaned to prose": doc is not None
        and "<p>" not in doc["text"]
        and "Small Model" in doc["text"],
        "manifest written": os.path.exists(manifest) and "local/news-firehose" in manifest_text,
        "absolute manifest path is data-relative": "data_files: "
        + os.path.relpath(out, os.path.join(ROOT, "data"))
        in manifest_text,
        "relative data path is not double-prefixed": "data_files: news/news.jsonl"
        in rel_manifest_text,
    }
    ok = True
    for name, passed in checks.items():
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\nSELFTEST", "PASS - RSS freshness docs are dataprep-ready" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="RSS/Atom news ingest for CHARKHA freshness replay")
    ap.add_argument("--feeds", help="text file of RSS/Atom feed URLs")
    ap.add_argument("--out", help="append JSONL docs here")
    ap.add_argument("--manifest", help="write dataprep local manifest here")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--since-hours", type=float, default=None)
    ap.add_argument("--no-fetch-pages", action="store_true", help="use feed summaries only")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.feeds or not args.out:
        ap.error("--feeds and --out are required")
    ingest(
        args.feeds,
        args.out,
        limit=args.limit,
        since_hours=args.since_hours,
        fetch_pages=not args.no_fetch_pages,
        manifest_path=args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
