#!/usr/bin/env python3
"""Re-status every URL the submission names. A link verified on the day it was pasted proves nothing.

Run: python scripts/check_links.py [post/post.md]
"""
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (link-check; so-gio)"}


def check(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read(4000).decode("utf-8", "ignore")
            return r.status, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, url
    except Exception as e:                     # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "post" / "post.md"
    text = src.read_text(encoding="utf-8")
    urls = sorted(set(re.findall(r"https?://[^\s)\"'<>]+", text)))
    bad = 0
    for url in urls:
        code, final = check(url.rstrip(".,;"))
        ok = code == 200
        bad += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'} {code:>3} {url[:88]}" + ("" if final == url else f" -> {final[:60]}"))
    print(f"\n{len(urls) - bad}/{len(urls)} reachable")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
