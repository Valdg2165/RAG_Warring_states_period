"""
Wikipedia crawler for the Warring States period of China.
- Starts from a set of seed URLs
- Follows internal Wikipedia links staying within the topic
- Extracts clean text using trafilatura
- Saves results to data/crawler_output.jsonl
"""

import json
import time
import re
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
import trafilatura

# ── Configuration ──────────────────────────────────────────────────────────────

SEED_URLS = [
    "https://en.wikipedia.org/wiki/Warring_States_period",
    "https://en.wikipedia.org/wiki/Seven_Warring_States",
    "https://en.wikipedia.org/wiki/State_of_Qin",
    "https://en.wikipedia.org/wiki/State_of_Chu",
    "https://en.wikipedia.org/wiki/State_of_Wei",
    "https://en.wikipedia.org/wiki/State_of_Zhao",
    "https://en.wikipedia.org/wiki/State_of_Han_(state)",
    "https://en.wikipedia.org/wiki/State_of_Yan",
    "https://en.wikipedia.org/wiki/State_of_Qi",
    "https://en.wikipedia.org/wiki/Shang_Yang",
    "https://en.wikipedia.org/wiki/Sun_Tzu",
    "https://en.wikipedia.org/wiki/Confucius",
    "https://en.wikipedia.org/wiki/Mencius",
    "https://en.wikipedia.org/wiki/Legalism_(Chinese_philosophy)",
    "https://en.wikipedia.org/wiki/Hundred_Schools_of_Thought",
]

# Only follow links that match these patterns (keep on-topic)
TOPIC_PATTERNS = [
    r"/wiki/Warring_States",
    r"/wiki/State_of_",
    r"/wiki/Battle_of_",
    r"/wiki/Zhou_dynasty",
    r"/wiki/Qin_dynasty",
    r"/wiki/Shang_Yang",
    r"/wiki/Sun_Tzu",
    r"/wiki/Confucius",
    r"/wiki/Mencius",
    r"/wiki/Laozi",
    r"/wiki/Zhuangzi",
    r"/wiki/Han_Fei",
    r"/wiki/Li_Si",
    r"/wiki/Bai_Qi",
    r"/wiki/Wu_Qi",
    r"/wiki/Legalism",
    r"/wiki/Confucianism",
    r"/wiki/Taoism",
    r"/wiki/Mohism",
    r"/wiki/Hundred_Schools",
    r"/wiki/Spring_and_Autumn",
    r"/wiki/Qin_Shi_Huang",
    r"/wiki/Zhao_Zhengping",
    r"/wiki/Lord_Shang",
    r"/wiki/Stratagem",
    r"/wiki/Chinese_philosophy",
]

OUTPUT_FILE = "data/crawler_output.jsonl"
MAX_PAGES = 20          # crawl up to this many pages
MIN_WORDS = 300         # skip pages shorter than this
DELAY = 1.0             # seconds between requests (be polite)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WarringStatesCrawler/1.0; "
                  "educational research project)"
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_valid_wiki_link(href: str) -> bool:
    """Return True if href is a Wikipedia article link (not a special page)."""
    if not href.startswith("/wiki/"):
        return False
    skip = (
        ":", "File:", "Help:", "Wikipedia:", "Talk:", "User:", "Category:",
        "Template:", "Special:", "Portal:", "Main_Page"
    )
    return not any(s in href for s in skip)


def is_on_topic(href: str) -> bool:
    return any(re.search(p, href) for p in TOPIC_PATTERNS)


def extract_links(html: str, base_url: str) -> list[str]:
    """Pull all on-topic Wikipedia article links from raw HTML."""
    links = []
    for match in re.finditer(r'href="(/wiki/[^"#?]+)"', html):
        href = match.group(1)
        if is_valid_wiki_link(href) and is_on_topic(href):
            links.append(urljoin("https://en.wikipedia.org", href))
    return list(set(links))


# Wikipedia footer section headings — everything from these onwards is discarded
FOOTER_HEADINGS = re.compile(
    r"\n(See also|References|External links|Notes|Further reading|"
    r"Bibliography|Footnotes|Citations|Sources|Works cited)\s*\n",
    re.IGNORECASE,
)


def strip_footer(text: str) -> str:
    """Cut Wikipedia text at the first footer section heading."""
    m = FOOTER_HEADINGS.search(text)
    if m:
        return text[:m.start()].strip()
    return text


def fetch_and_extract(url: str, session: requests.Session) -> dict | None:
    """Fetch a URL, extract main text, return dict or None if unusable."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  [skip] HTTP {resp.status_code}: {url}")
            return None
        html = resp.text
    except Exception as e:
        print(f"  [error] {e}: {url}")
        return None

    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:
        return None

    text = strip_footer(text)   # remove References / External links / etc.

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        print(f"  [skip] too short ({word_count} words): {url}")
        return None

    title = re.search(r'<title>(.*?) - Wikipedia</title>', html)
    title = title.group(1) if title else url.split("/wiki/")[-1].replace("_", " ")

    return {
        "url": url,
        "title": title,
        "text": text,
        "word_count": word_count,
        "links_found": extract_links(html, url),
    }


# ── Main crawl loop ─────────────────────────────────────────────────────────────

def crawl():
    visited: set[str] = set()
    queue: deque[str] = deque(SEED_URLS)
    results: list[dict] = []

    session = requests.Session()

    print(f"Starting crawl (max {MAX_PAGES} pages, min {MIN_WORDS} words)\n")

    while queue and len(results) < MAX_PAGES:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(results)+1}/{MAX_PAGES}] {url}")
        data = fetch_and_extract(url, session)

        if data:
            print(f"  OK — {data['word_count']} words, "
                  f"{len(data['links_found'])} on-topic links found")
            results.append(data)
            # Enqueue new links (not yet visited)
            for link in data["links_found"]:
                if link not in visited:
                    queue.append(link)

        time.sleep(DELAY)

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            # Don't store links_found in the output file (not needed downstream)
            out = {k: v for k, v in item.items() if k != "links_found"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)} pages saved to {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    crawl()
