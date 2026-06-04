"""
Tamil Wikipedia Data Downloader
---------------------------------
Two modes:

  python tamil_wikipedia.py --api          # fetch via API (~10-20MB, easy)
  python tamil_wikipedia.py --dump         # full dump (~2GB text, serious training)
  python tamil_wikipedia.py --api --limit 500   # fetch 500 articles via API

Output: data/tamil_wiki.txt  (appended to data/tamil.txt for training)

API mode:   no extra dependencies
Dump mode:  pip install wikiextractor
"""

import os, sys, json, time, re, urllib.request, urllib.parse

DATA_DIR   = 'data'
WIKI_PATH  = os.path.join(DATA_DIR, 'tamil_wiki.txt')
TAMIL_PATH = os.path.join(DATA_DIR, 'tamil.txt')

# Tamil Wikipedia API endpoint
API_URL = "https://ta.wikipedia.org/w/api.php"

# Full dump URL (Tamil Wikipedia, latest articles)
DUMP_URL = "https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2"


# ── API mode ──────────────────────────────────────────────────────────────────

def clean_wiki_text(text):
    """Remove Wikipedia markup, keep clean Tamil text."""
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)  # [[link|text]] → text
    text = re.sub(r'\{\{[^}]*\}\}', '', text)    # remove {{templates}}
    text = re.sub(r'<[^>]+>', '', text)           # remove HTML tags
    text = re.sub(r'={2,}[^=]+=+', '', text)      # remove == headings ==
    text = re.sub(r'\[\d+\]', '', text)           # remove [1] citations
    text = re.sub(r'https?://\S+', '', text)      # remove URLs
    text = re.sub(r'[ \t]+', ' ', text)           # collapse spaces
    text = re.sub(r'\n{3,}', '\n\n', text)        # collapse blank lines
    # keep only Tamil + basic punctuation
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if (0x0B80 <= cp <= 0x0BFF) or ch in ' \n.,!?;:\'"()–-൦൧൨൩൪':
            cleaned.append(ch)
    return ''.join(cleaned).strip()


def api_fetch_random_articles(n=200):
    """Fetch n random Tamil Wikipedia articles via the MediaWiki API."""
    os.makedirs(DATA_DIR, exist_ok=True)
    collected = []
    total_chars = 0
    batch = 20  # fetch 20 random titles at a time

    print(f"Fetching {n} Tamil Wikipedia articles via API...")

    while len(collected) < n:
        # step 1: get random page titles
        params = urllib.parse.urlencode({
            'action': 'query',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': batch,
            'format': 'json',
        })
        try:
            with urllib.request.urlopen(f"{API_URL}?{params}", timeout=10) as r:
                data = json.loads(r.read().decode('utf-8'))
            titles = [p['title'] for p in data['query']['random']]
        except Exception as e:
            print(f"  API error (titles): {e}")
            time.sleep(2)
            continue

        # step 2: fetch content for those titles
        params2 = urllib.parse.urlencode({
            'action': 'query',
            'titles': '|'.join(titles),
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain',
            'format': 'json',
        })
        try:
            with urllib.request.urlopen(f"{API_URL}?{params2}", timeout=15) as r:
                data2 = json.loads(r.read().decode('utf-8'))
            pages = data2['query']['pages']
        except Exception as e:
            print(f"  API error (content): {e}")
            time.sleep(2)
            continue

        for page in pages.values():
            text = page.get('extract', '')
            if not text:
                continue
            clean = clean_wiki_text(text)
            # only keep articles with substantial Tamil content
            tamil_chars = sum(1 for c in clean if 0x0B80 <= ord(c) <= 0x0BFF)
            if tamil_chars < 100:
                continue
            collected.append(clean)
            total_chars += len(clean)

        done = min(len(collected), n)
        print(f"  {done}/{n} articles  |  {total_chars:,} Tamil chars", end='\r')
        time.sleep(0.5)   # be polite to Wikipedia's servers

    print(f"\nDone — {len(collected)} articles, {total_chars:,} characters")

    full_text = '\n\n'.join(collected[:n])
    with open(WIKI_PATH, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"Saved to {WIKI_PATH}")
    return full_text


# ── Dump mode ─────────────────────────────────────────────────────────────────

def download_dump():
    """Download Tamil Wikipedia XML dump and extract clean text using wikiextractor."""
    try:
        import wikiextractor  # noqa
    except ImportError:
        print("wikiextractor not installed.")
        print("Run: pip install wikiextractor")
        sys.exit(1)

    dump_path = os.path.join(DATA_DIR, 'tawiki-latest.xml.bz2')
    extract_dir = os.path.join(DATA_DIR, 'wiki_extracted')

    os.makedirs(DATA_DIR, exist_ok=True)

    # download
    if not os.path.exists(dump_path):
        print(f"Downloading Tamil Wikipedia dump (~300-500MB)...")
        print(f"URL: {DUMP_URL}")
        print("This will take a few minutes depending on your connection...")

        def progress(count, block_size, total_size):
            pct = count * block_size / total_size * 100
            mb  = count * block_size / 1024 / 1024
            print(f"\r  {pct:.1f}%  {mb:.0f}MB", end='', flush=True)

        urllib.request.urlretrieve(DUMP_URL, dump_path, reporthook=progress)
        print(f"\nDownloaded to {dump_path}")
    else:
        print(f"Dump already exists: {dump_path}")

    # extract using wikiextractor
    print("Extracting text from dump (takes 5-15 mins)...")
    os.system(f"python -m wikiextractor.WikiExtractor {dump_path} "
              f"--output {extract_dir} --bytes 10M --quiet")

    # combine all extracted files into one Tamil text file
    print("Combining extracted files...")
    all_text = []
    for root, _, files in os.walk(extract_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            with open(path, encoding='utf-8') as f:
                raw = f.read()
            # wikiextractor wraps articles in <doc>...</doc>
            articles = re.findall(r'<doc[^>]*>(.*?)</doc>', raw, re.DOTALL)
            for article in articles:
                clean = clean_wiki_text(article)
                tamil_chars = sum(1 for c in clean if 0x0B80 <= ord(c) <= 0x0BFF)
                if tamil_chars > 200:
                    all_text.append(clean)

    full_text = '\n\n'.join(all_text)
    with open(WIKI_PATH, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"Saved {len(all_text):,} articles ({len(full_text):,} chars) to {WIKI_PATH}")
    return full_text


# ── Merge into main training file ─────────────────────────────────────────────

def merge_into_training_data():
    """Append Tamil Wikipedia text to data/tamil.txt for training."""
    if not os.path.exists(WIKI_PATH):
        print("No wiki data found. Run with --api or --dump first.")
        return

    with open(WIKI_PATH, encoding='utf-8') as f:
        wiki_text = f.read()

    existing = ''
    if os.path.exists(TAMIL_PATH):
        with open(TAMIL_PATH, encoding='utf-8') as f:
            existing = f.read()

    combined = existing + '\n\n# தமிழ் விக்கிபீடியா (Tamil Wikipedia)\n\n' + wiki_text

    with open(TAMIL_PATH, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"\nMerged into {TAMIL_PATH}")
    print(f"  Total size: {len(combined):,} characters")
    tamil_chars = sum(1 for c in combined if 0x0B80 <= ord(c) <= 0x0BFF)
    print(f"  Tamil chars: {tamil_chars:,}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = sys.argv[1:]

    if '--dump' in args:
        download_dump()
        merge_into_training_data()

    elif '--api' in args:
        limit = int(args[args.index('--limit') + 1]) if '--limit' in args else 200
        api_fetch_random_articles(limit)
        merge_into_training_data()

    else:
        print("Usage:")
        print("  python tamil_wikipedia.py --api            # fetch 200 articles (~10MB, easy)")
        print("  python tamil_wikipedia.py --api --limit 500  # fetch 500 articles")
        print("  python tamil_wikipedia.py --dump           # full dump (~2GB, serious training)")
