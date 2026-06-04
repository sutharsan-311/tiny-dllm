"""
Tamil Dataset Loader
---------------------
Downloads and prepares Tamil text for training:
  1. Thirukkural — all 1330 couplets (classical Tamil, ~2000 years old)
  2. Project Madurai — ancient Sangam literature (optional, larger)

Usage:
  python tamil_dataset.py           # download + save to data/tamil.txt
  python tamil_dataset.py --stats   # show dataset stats
"""

import os, json, urllib.request, sys

DATA_DIR  = 'data'
TAMIL_PATH = os.path.join(DATA_DIR, 'tamil.txt')

# Thirukkural JSON — all 1330 couplets with Tamil text
KURAL_URL = "https://raw.githubusercontent.com/tk120404/thirukkural/master/data/thirukkural.json"


def download_thirukkural():
    """Download all 1330 Thirukkural couplets and return as Tamil text."""
    print("Downloading Thirukkural...")
    try:
        with urllib.request.urlopen(KURAL_URL, timeout=15) as r:
            data = json.loads(r.read().decode('utf-8'))
    except Exception as e:
        print(f"  Failed to download: {e}")
        return ""

    lines = []
    for entry in data:
        # each entry has 'l1' and 'l2' — the two lines of the kural in Tamil
        l1 = entry.get('l1', '').strip()
        l2 = entry.get('l2', '').strip()
        if l1 and l2:
            lines.append(l1)
            lines.append(l2)
            lines.append('')   # blank line between couplets

    text = '\n'.join(lines)
    print(f"  Got {len([e for e in data])} kurals ({len(text):,} characters)")
    return text


def download_project_madurai():
    """
    A small selection of Sangam poetry from Project Madurai.
    Project Madurai (projectmadurai.org) hosts ancient Tamil texts freely.
    We use a small sample here — extend with more texts as needed.
    """
    # Natrinai — one of the Eight Anthologies (Ettuthokai), Sangam era
    URL = "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0002_1.html"
    print("Downloading Sangam poetry (Natrinai)...")
    try:
        req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode('utf-8', errors='ignore')
        # extract text between <body> tags, strip HTML
        import re
        raw = re.sub(r'<[^>]+>', ' ', raw)       # remove HTML tags
        raw = re.sub(r'[ \t]+', ' ', raw)         # collapse spaces
        raw = re.sub(r'\n{3,}', '\n\n', raw)      # collapse blank lines
        # keep only Tamil unicode block + basic punctuation
        cleaned = []
        for ch in raw:
            cp = ord(ch)
            if (0x0B80 <= cp <= 0x0BFF) or ch in ' \n.,!?;:\'"()–-':
                cleaned.append(ch)
        text = ''.join(cleaned).strip()
        print(f"  Got {len(text):,} Tamil characters from Natrinai")
        return text
    except Exception as e:
        print(f"  Skipping Sangam poetry (network issue): {e}")
        return ""


def build_dataset(include_madurai=False):
    os.makedirs(DATA_DIR, exist_ok=True)

    parts = []

    kural = download_thirukkural()
    if kural:
        parts.append("# திருக்குறள் (Thirukkural)\n\n" + kural)

    if include_madurai:
        madurai = download_project_madurai()
        if madurai:
            parts.append("\n\n# நற்றிணை (Natrinai — Sangam Poetry)\n\n" + madurai)

    if not parts:
        print("No data downloaded. Check your internet connection.")
        return

    full_text = '\n\n'.join(parts)

    with open(TAMIL_PATH, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"\nSaved to {TAMIL_PATH}")
    stats(full_text)


def stats(text=None):
    if text is None:
        if not os.path.exists(TAMIL_PATH):
            print("No dataset found. Run: python tamil_dataset.py")
            return
        with open(TAMIL_PATH, encoding='utf-8') as f:
            text = f.read()

    # Tamil Unicode block: U+0B80–U+0BFF
    tamil_chars = [c for c in text if 0x0B80 <= ord(c) <= 0x0BFF]
    unique_chars = sorted(set(text))
    tamil_unique = [c for c in unique_chars if 0x0B80 <= ord(c) <= 0x0BFF]

    print(f"\nDataset stats:")
    print(f"  Total characters  : {len(text):,}")
    print(f"  Tamil characters  : {len(tamil_chars):,} ({len(tamil_chars)/len(text)*100:.1f}%)")
    print(f"  Unique characters : {len(unique_chars)}")
    print(f"  Tamil unique chars: {len(tamil_unique)}")
    print(f"  Sample:")
    # find first Tamil line
    for line in text.splitlines():
        if any(0x0B80 <= ord(c) <= 0x0BFF for c in line) and len(line) > 5:
            print(f"    {line}")
            break


if __name__ == '__main__':
    if '--stats' in sys.argv:
        stats()
    else:
        include_madurai = '--madurai' in sys.argv
        build_dataset(include_madurai)
