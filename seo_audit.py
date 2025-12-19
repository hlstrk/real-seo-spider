from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    import config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore


GENERIC_ALT_RE = re.compile(
    r"^\s*(image|photo|picture|img|logo|banner|icon|slider|placeholder)\s*$",
    re.IGNORECASE,
)
GENERIC_FILENAME_RE = re.compile(
    r"^(img|image|photo|picture|banner|slider|icon)[-_]?\d*$", re.IGNORECASE
)
CAMERA_FILENAME_RE = re.compile(r"^(img|dsc|pxl|screenshot)[-_]?\d+", re.IGNORECASE)
ALT_BAD_REGEX = re.compile(
    r"(adsiz|adsız|proje|şablon|sablon|template|ornek|örnek|example|chatgpt|openai|jjjj+|kkkk+|\(\d+\))",
    re.IGNORECASE,
)
SITEMAP_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
PLACEHOLDER_IMG_RE = re.compile(r"/blank\\.gif($|\\?)", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".svg", ".bmp", ".ico"}


def _strip_query_fragment(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    u = u.split("#", 1)[0]
    u = u.split("?", 1)[0]
    return u


def is_probable_image_asset_url(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    if url.startswith(("data:", "data://")):
        return True
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    last = (parsed.path or "").split("/")[-1]
    if not last or "." not in last:
        return False
    ext = os.path.splitext(last)[1].lower()
    return ext in IMAGE_EXTS


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class PageImage:
    src: str
    filename: str
    alt: str


@dataclass(frozen=True)
class RenderedImage:
    src: str
    current_src: str
    alt: str
    loading: str
    decoding: str
    fetchpriority: str
    srcset: str
    sizes: str
    x: float
    y: float
    width: float
    height: float
    natural_width: int
    natural_height: int
    in_viewport: bool


@dataclass(frozen=True)
class PageData:
    url: str
    final_url: str
    status_code: int
    content_type: str
    hsts: str
    content_hash: str
    content_text_len: int
    html_lang: str
    viewport_width: int
    viewport_height: int
    title: str
    meta_description: str
    h1: str
    canonical: str
    preload_images: Tuple[str, ...]
    images: Tuple[PageImage, ...]
    rendered_images: Tuple[RenderedImage, ...]


@dataclass(frozen=True)
class Issue:
    url: str
    issue_type: str
    field: str
    severity: str
    message: str
    suggestion: str
    category: str = "general"
    evidence: str = ""
    source: str = "local"


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    headers: Dict[str, str] = dataclasses.field(default_factory=dict)
    html: str = ""
    viewport_width: int = 0
    viewport_height: int = 0
    rendered_images: Tuple[Dict[str, Any], ...] = tuple()
    error: str = ""
    redirect_location: str = ""


def _get_cfg(name: str, default: Any) -> Any:
    if config is not None and hasattr(config, name):
        value = getattr(config, name)
        if value not in (None, ""):
            return value
    return os.getenv(name, default)


def _get_list_cfg(name: str, default: List[str]) -> List[str]:
    if config is not None and hasattr(config, name):
        value = getattr(config, name)
        if isinstance(value, (list, tuple)):
            items = [str(v).strip() for v in value if str(v).strip()]
            return items if items else default
        if isinstance(value, str) and value.strip():
            items = [v.strip() for v in value.split(",") if v.strip()]
            return items if items else default
    env_val = os.getenv(name, "").strip()
    if env_val:
        items = [v.strip() for v in env_val.split(",") if v.strip()]
        return items if items else default
    return default


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    if not prefix:
        return "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if prefix != "/" and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def normalize_page_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith(("data:", "data://", "mailto:", "tel:", "javascript:", "blob:")):
        return raw

    parsed = urlparse(raw)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc and parsed.path:
        parsed2 = urlparse(f"{scheme}://{raw}")
        scheme = parsed2.scheme
        netloc = parsed2.netloc
        parsed = parsed2

    cleaned = ParseResult(
        scheme=scheme.lower(),
        netloc=netloc.lower(),
        path=parsed.path or "/",
        params="",
        query="",
        fragment="",
    )
    if cleaned.path != "/" and not cleaned.path.endswith("/") and "." not in cleaned.path.split("/")[-1]:
        cleaned = cleaned._replace(path=cleaned.path + "/")
    return urlunparse(cleaned)


def normalize_asset_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""

    # data: URLs can be very large; keep them intact and do not try to "normalize" them.
    if raw.startswith("data:"):
        return raw
    if raw.startswith("data://") and raw[7:].startswith("data:"):
        return raw[7:]

    parsed = urlparse(raw)
    if not parsed.scheme and not parsed.netloc:
        return raw

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc and parsed.path:
        parsed2 = urlparse(f"{scheme}://{raw}")
        scheme = parsed2.scheme
        netloc = parsed2.netloc
        parsed = parsed2

    cleaned = ParseResult(
        scheme=scheme.lower(),
        netloc=netloc.lower(),
        path=parsed.path or "",
        params="",
        query=parsed.query or "",
        fragment="",
    )
    return urlunparse(cleaned)


# Backward-compat for older code paths: normalize_url means "page URL".
normalize_url = normalize_page_url


def is_in_scope(url: str, base_domain: str, scope_prefixes: List[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    base_domain = base_domain.lower()
    base_domain = base_domain[4:] if base_domain.startswith("www.") else base_domain
    if host != base_domain:
        return False
    prefixes = [_normalize_prefix(p) for p in (scope_prefixes or ["/"])]
    path = parsed.path or "/"
    for pfx in prefixes:
        if pfx == "/":
            return True
        if path == pfx[:-1] or path.startswith(pfx):
            return True
    return False


def parse_page_data(fetch: FetchResult) -> PageData:
    final_url = normalize_page_url(fetch.final_url)
    soup = BeautifulSoup(fetch.html or "", "html.parser")

    hsts = ""
    try:
        hsts = str((fetch.headers or {}).get("strict-transport-security") or "").strip()
    except Exception:
        hsts = ""

    # Duplicate content detection (lightweight): hash visible text.
    content_text_len = 0
    content_hash = ""
    try:
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        root = soup.find("main") or soup.body or soup
        text = root.get_text(" ", strip=True) if root else ""
        text = re.sub(r"\\s+", " ", (text or "")).strip().lower()
        content_text_len = len(text)
        if content_text_len >= 300:
            # Limit to keep stable and small.
            sample = text[:8000].encode("utf-8", errors="ignore")
            content_hash = hashlib.sha1(sample).hexdigest()
    except Exception:
        content_text_len = 0
        content_hash = ""

    html_lang = ""
    try:
        if soup.html and soup.html.has_attr("lang"):
            html_lang = str(soup.html.get("lang") or "").strip()
    except Exception:
        html_lang = ""

    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    meta_desc_el = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    meta_description = (meta_desc_el.get("content") if meta_desc_el else "") or ""
    meta_description = meta_description.strip()

    h1_el = soup.find("h1")
    h1 = (h1_el.get_text(" ", strip=True) if h1_el else "").strip()

    canonical_el = soup.find("link", rel=lambda v: v and "canonical" in v)
    canonical = (canonical_el.get("href") if canonical_el else "") or ""
    canonical = canonical.strip()

    preload_images: List[str] = []
    for link in soup.find_all("link"):
        try:
            rel = link.get("rel") or []
            if isinstance(rel, str):
                rel_list = [rel]
            else:
                rel_list = [str(r) for r in rel]
            rel_list = [r.lower() for r in rel_list]
            if "preload" not in rel_list:
                continue
            if str(link.get("as") or "").lower() != "image":
                continue
            href = str(link.get("href") or "").strip()
            if not href:
                continue
            preload_images.append(normalize_asset_url(urljoin(final_url, href)))
        except Exception:
            continue

    def _pick_img_src(tag) -> str:
        candidates: List[Tuple[str, str]] = []
        for attr in ("data-src", "data-lazy-src", "data-original", "data-srcset", "srcset", "src"):
            val = (tag.get(attr) or "").strip()
            if val:
                candidates.append((attr, val))
        if not candidates:
            return ""

        def first_url(srcset: str) -> str:
            first = srcset.split(",")[0].strip()
            return first.split(" ")[0].strip()

        data_src = ""
        src = ""
        srcset = ""
        data_srcset = ""
        for attr, val in candidates:
            if attr in ("data-src", "data-lazy-src", "data-original"):
                data_src = data_src or val
            elif attr == "src":
                src = src or val
            elif attr == "srcset":
                srcset = srcset or val
            elif attr == "data-srcset":
                data_srcset = data_srcset or val

        if src and PLACEHOLDER_IMG_RE.search(src):
            if data_src:
                return data_src
            if data_srcset:
                return first_url(data_srcset)
            if srcset:
                return first_url(srcset)
            return src

        if data_src:
            return data_src
        if src:
            return src
        if srcset:
            return first_url(srcset)
        if data_srcset:
            return first_url(data_srcset)
        return ""

    images: List[PageImage] = []
    for img in soup.find_all("img"):
        src = _pick_img_src(img)
        if not src:
            continue
        # Ignore Polylang / language switcher flags embedded as base64 SVG (very noisy and not SEO-critical).
        if src.startswith("data:image/svg+xml;base64,") or src.startswith("data://data:image/svg+xml;base64,"):
            continue
        src_abs = normalize_asset_url(urljoin(final_url, src))
        filename = urlparse(src_abs).path.split("/")[-1]
        alt = (img.get("alt") or "").strip()
        images.append(PageImage(src=src_abs, filename=filename, alt=alt))

    rendered: List[RenderedImage] = []
    for r in fetch.rendered_images or ():
        try:
            src_raw = str(r.get("src") or "").strip()
            current_raw = str(r.get("current_src") or r.get("currentSrc") or "").strip()
            rendered.append(
                RenderedImage(
                    src=normalize_asset_url(urljoin(final_url, src_raw)) if src_raw else "",
                    current_src=normalize_asset_url(urljoin(final_url, current_raw)) if current_raw else "",
                    alt=str(r.get("alt") or "").strip(),
                    loading=str(r.get("loading") or "").strip(),
                    decoding=str(r.get("decoding") or "").strip(),
                    fetchpriority=str(r.get("fetchpriority") or r.get("fetchPriority") or "").strip(),
                    srcset=str(r.get("srcset") or "").strip(),
                    sizes=str(r.get("sizes") or "").strip(),
                    x=float(r.get("x") or 0),
                    y=float(r.get("y") or 0),
                    width=float(r.get("width") or 0),
                    height=float(r.get("height") or 0),
                    natural_width=int(r.get("natural_width") or r.get("naturalWidth") or 0),
                    natural_height=int(r.get("natural_height") or r.get("naturalHeight") or 0),
                    in_viewport=bool(r.get("in_viewport") or r.get("inViewport") or False),
                )
            )
        except Exception:
            continue

    return PageData(
        url=normalize_page_url(fetch.requested_url),
        final_url=final_url,
        status_code=fetch.status_code,
        content_type=(fetch.content_type or "").strip(),
        hsts=hsts,
        content_hash=content_hash,
        content_text_len=int(content_text_len or 0),
        html_lang=html_lang,
        viewport_width=int(fetch.viewport_width or 0),
        viewport_height=int(fetch.viewport_height or 0),
        title=title,
        meta_description=meta_description,
        h1=h1,
        canonical=canonical,
        preload_images=tuple(dict.fromkeys(preload_images)),
        images=tuple(images),
        rendered_images=tuple(rendered),
    )


def extract_links(html: str, base_url: str) -> Iterable[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = href.strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        yield normalize_url(urljoin(base_url, href))


# === FETCHERS / CRAWL / AUDIT SECTIONS BELOW ===


class RequestsFetcher:
    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds
        self._headers = {
            "User-Agent": "real-seo-spider/1.0 (+https://example.local)",
            "Accept": "text/html,application/xhtml+xml",
        }

    def fetch(self, url: str, allow_redirects: bool = True) -> FetchResult:
        url = normalize_url(url)
        try:
            resp = requests.get(
                url, headers=self._headers, timeout=self._timeout_seconds, allow_redirects=allow_redirects
            )
            final_url = normalize_url(resp.url)
            hdrs = {
                "strict-transport-security": str(resp.headers.get("Strict-Transport-Security") or "").strip(),
            }
            return FetchResult(
                requested_url=url,
                final_url=final_url,
                status_code=int(resp.status_code),
                content_type=str(resp.headers.get("Content-Type") or ""),
                headers=hdrs,
                html=str(resp.text or ""),
                redirect_location=str(resp.headers.get("Location") or "") if not allow_redirects else "",
            )
        except requests.RequestException as e:
            return FetchResult(
                requested_url=url,
                final_url=url,
                status_code=0,
                content_type="",
                headers={},
                html="",
                error=str(e),
            )


class ChromiumFetcher:
    def __init__(self, timeout_seconds: int, headless: bool = True) -> None:
        self._timeout_ms = int(timeout_seconds * 1000)
        self._headless = headless
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def __enter__(self) -> "ChromiumFetcher":
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Playwright is not available. Run: python -m pip install playwright "
                "and then: python -m playwright install chromium"
            ) from e

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._context = self._browser.new_context(
            locale="en-US",
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
        )
        self._page = self._context.new_page()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            try:
                if self._browser is not None:
                    self._browser.close()
            finally:
                if self._pw is not None:
                    self._pw.stop()

    def fetch(self, url: str, allow_redirects: bool = True) -> FetchResult:
        url = normalize_url(url)
        if not allow_redirects:
            return FetchResult(
                requested_url=url,
                final_url=url,
                status_code=0,
                content_type="",
                html="",
                error="chromium fetcher does not support allow_redirects=False (use requests fetcher).",
            )
        try:
            resp = self._page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
            try:
                self._page.wait_for_load_state("networkidle", timeout=min(5000, self._timeout_ms))
            except Exception:
                pass

            # Trigger lazy-loaded images by scrolling a bit.
            try:
                vh = self._page.evaluate("() => window.innerHeight || 900")
                for y in (0, int(vh), int(vh) * 2, int(vh) * 3):
                    self._page.evaluate("(yy) => window.scrollTo(0, yy)", y)
                    self._page.wait_for_timeout(250)
                self._page.evaluate("() => window.scrollTo(0, 0)")
                self._page.wait_for_timeout(200)
            except Exception:
                pass

            html = self._page.content()
            final_url = normalize_url(self._page.url)
            headers = resp.headers if resp is not None else {}
            content_type = str(headers.get("content-type") or "")
            status_code = int(resp.status) if resp is not None else 0
            hdrs = {
                "strict-transport-security": str(headers.get("strict-transport-security") or "").strip(),
            }

            render_payload = {"viewport": {"w": 0, "h": 0}, "images": []}
            try:
                render_payload = self._page.evaluate(
                    """() => {
                      const v = { w: window.innerWidth || 0, h: window.innerHeight || 0 };
                      const imgs = Array.from(document.querySelectorAll('img')).map((el) => {
                        const r = el.getBoundingClientRect();
                        const cur = (el.currentSrc || el.src || '').trim();
                        const ds = (el.getAttribute('data-src') || '').trim();
                        const dls = (el.getAttribute('data-lazy-src') || '').trim();
                        const dor = (el.getAttribute('data-original') || '').trim();
                        const placeholder = (el.getAttribute('src') || '').includes('/blank.gif');
                        const best = (placeholder && (ds || dls || dor)) ? (ds || dls || dor) : cur;
                        return {
                          src: el.getAttribute('src') || '',
                          current_src: best || cur,
                          alt: el.getAttribute('alt') || '',
                          loading: el.getAttribute('loading') || '',
                          decoding: el.getAttribute('decoding') || '',
                          fetchpriority: el.getAttribute('fetchpriority') || '',
                          srcset: el.getAttribute('srcset') || '',
                          sizes: el.getAttribute('sizes') || '',
                          x: r.left || 0,
                          y: r.top || 0,
                          width: r.width || 0,
                          height: r.height || 0,
                          natural_width: el.naturalWidth || 0,
                          natural_height: el.naturalHeight || 0,
                          in_viewport: (r.bottom > 0 && r.right > 0 && r.top < v.h && r.left < v.w),
                        };
                      });
                      return { viewport: v, images: imgs };
                    }"""
                )
            except Exception:
                render_payload = {"viewport": {"w": 0, "h": 0}, "images": []}

            vw = int(((render_payload.get("viewport") or {}).get("w")) or 0)
            vh = int(((render_payload.get("viewport") or {}).get("h")) or 0)
            imgs = render_payload.get("images") or []

            return FetchResult(
                requested_url=url,
                final_url=final_url,
                status_code=status_code,
                content_type=content_type,
                headers=hdrs,
                html=html,
                viewport_width=vw,
                viewport_height=vh,
                rendered_images=tuple(imgs) if isinstance(imgs, list) else tuple(),
            )
        except Exception as e:
            final_url = normalize_url(getattr(self._page, "url", url) or url)
            return FetchResult(
                requested_url=url,
                final_url=final_url,
                status_code=0,
                content_type="",
                headers={},
                html="",
                error=str(e),
            )


def discover_urls_from_sitemap(
    sitemap_url: str,
    base_domain: str,
    scope_prefixes: List[str],
    timeout_seconds: int,
    max_sitemaps: int = 50,
) -> List[str]:
    sitemap_url = normalize_url(sitemap_url)
    seen: Set[str] = set()
    queue: Deque[str] = deque([sitemap_url])
    found: Set[str] = set()

    while queue and len(seen) < max_sitemaps:
        sm = queue.popleft()
        if sm in seen:
            continue
        seen.add(sm)

        try:
            resp = requests.get(
                sm,
                headers={"User-Agent": "real-seo-spider/1.0 (+https://example.local)"},
                timeout=timeout_seconds,
                allow_redirects=True,
            )
        except requests.RequestException:
            continue

        if resp.status_code >= 400:
            continue

        for loc in SITEMAP_LOC_RE.findall(resp.text or ""):
            loc = normalize_url(loc.strip())
            if loc.endswith(".xml"):
                queue.append(loc)
            elif is_in_scope(loc, base_domain, scope_prefixes):
                found.add(loc)

    return sorted(found)


def crawl(
    start_url: str,
    base_domain: str,
    scope_prefixes: List[str],
    max_pages: int,
    timeout_seconds: int,
    use_sitemap: bool = False,
    fetcher: str = "requests",
    chromium_headless: bool = True,
    require_lang: str = "",
) -> Tuple[List[PageData], List[Issue]]:
    start_url = normalize_url(start_url)
    scope_prefixes = [_normalize_prefix(p) for p in (scope_prefixes or ["/"])]
    require_lang = (require_lang or "").strip().lower()

    seed_issues: List[Issue] = []
    if not is_in_scope(start_url, base_domain, scope_prefixes) and not use_sitemap:
        raise ValueError(f"Start URL must be in scope (got {start_url}; prefixes={scope_prefixes})")

    visited: Set[str] = set()
    queue: Deque[str] = deque()
    results: List[PageData] = []

    if use_sitemap:
        seeds = discover_urls_from_sitemap(
            sitemap_url=f"https://{base_domain}/sitemap.xml",
            base_domain=base_domain,
            scope_prefixes=scope_prefixes,
            timeout_seconds=timeout_seconds,
        )
        for u in seeds[: max_pages * 3]:
            queue.append(u)
    else:
        rf = RequestsFetcher(timeout_seconds=timeout_seconds)
        start_fetch = rf.fetch(start_url, allow_redirects=True)
        if start_fetch.error:
            seed_issues.append(
                Issue(
                    url=start_url,
                    issue_type="request_failed",
                    field="start_url",
                    severity="high",
                    message="Failed to fetch start URL.",
                    suggestion="Check connectivity and retry.",
                    category="crawl",
                    evidence=start_fetch.error,
                )
            )
            start_final = start_url
        else:
            start_final = normalize_url(start_fetch.final_url)

        if is_in_scope(start_final, base_domain, scope_prefixes):
            queue.append(start_final)
        else:
            seed_issues.append(
                Issue(
                    url=start_url,
                    issue_type="redirect_out_of_scope",
                    field="start_url",
                    severity="high",
                    message="Start URL redirected outside configured scope.",
                    suggestion="Fix redirect or adjust --scope-prefix / use --use-sitemap.",
                    category="crawl",
                    evidence=f"final={start_final}",
                )
            )
            seeds = discover_urls_from_sitemap(
                sitemap_url=f"https://{base_domain}/sitemap.xml",
                base_domain=base_domain,
                scope_prefixes=scope_prefixes,
                timeout_seconds=timeout_seconds,
            )
            for u in seeds[: max_pages * 3]:
                queue.append(u)

    if fetcher == "chromium":
        fetcher_ctx: Any = ChromiumFetcher(timeout_seconds=timeout_seconds, headless=chromium_headless)
    else:
        fetcher_ctx = None
        fetcher_obj = RequestsFetcher(timeout_seconds=timeout_seconds)

    def run_loop(fetcher_obj: Any) -> None:
        nonlocal results
        while queue and len(results) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            fr = fetcher_obj.fetch(url, allow_redirects=True)
            if fr.error:
                results.append(
                    PageData(
                        url=normalize_url(url),
                        final_url=normalize_url(fr.final_url or url),
                        status_code=0,
                        content_type="",
                        hsts="",
                        content_hash="",
                        content_text_len=0,
                        html_lang="",
                        viewport_width=0,
                        viewport_height=0,
                        title="",
                        meta_description="",
                        h1="",
                        canonical="",
                        preload_images=tuple(),
                        images=tuple(),
                        rendered_images=tuple(),
                    )
                )
                continue

            final_url = normalize_url(fr.final_url)
            if not is_in_scope(final_url, base_domain, scope_prefixes):
                continue

            page = parse_page_data(fr)
            if require_lang:
                if page.html_lang and not page.html_lang.lower().startswith(require_lang):
                    seed_issues.append(
                        Issue(
                            url=final_url,
                            issue_type="language_mismatch",
                            field="html_lang",
                            severity="medium",
                            message="Page language does not match required language.",
                            suggestion="Fix language attribute or adjust --require-lang.",
                            category="seo.meta",
                            evidence=f"lang={page.html_lang}",
                        )
                    )
                else:
                    results.append(page)
            else:
                results.append(page)

            if "text/html" not in (fr.content_type or "").lower():
                continue

            for link in extract_links(fr.html or "", final_url):
                if link not in visited and is_in_scope(link, base_domain, scope_prefixes):
                    queue.append(link)

    if fetcher == "chromium":
        with fetcher_ctx as ff:
            run_loop(ff)
    else:
        run_loop(fetcher_obj)

    return results, seed_issues


def audit_single(
    start_url: str,
    base_domain: str,
    scope_prefixes: List[str],
    timeout_seconds: int,
    follow_redirects: bool,
    fetcher: str = "requests",
    chromium_headless: bool = True,
    require_lang: str = "",
) -> Tuple[List[PageData], List[Issue]]:
    start_url = normalize_url(start_url)
    scope_prefixes = [_normalize_prefix(p) for p in (scope_prefixes or ["/"])]
    require_lang = (require_lang or "").strip().lower()
    issues: List[Issue] = []

    if fetcher == "chromium":
        with ChromiumFetcher(timeout_seconds=timeout_seconds, headless=chromium_headless) as cf:
            fr = cf.fetch(start_url, allow_redirects=True)
        if fr.error:
            return (
                [],
                [
                    Issue(
                        url=start_url,
                        issue_type="request_failed",
                        field="start_url",
                        severity="high",
                        message="Chromium navigation failed.",
                        suggestion="Check Playwright installation and retry.",
                        category="crawl",
                        evidence=fr.error,
                    )
                ],
            )
        final_url = normalize_url(fr.final_url)
        if not is_in_scope(final_url, base_domain, scope_prefixes):
            issues.append(
                Issue(
                    url=start_url,
                    issue_type="redirect_out_of_scope",
                    field="start_url",
                    severity="high",
                    message="Start URL ended outside configured scope after navigation.",
                    suggestion="Fix redirect or adjust --scope-prefix.",
                    category="crawl",
                    evidence=f"final={final_url}",
                )
            )
            return [], issues
        page = parse_page_data(fr)
        if require_lang and page.html_lang and not page.html_lang.lower().startswith(require_lang):
            issues.append(
                Issue(
                    url=final_url,
                    issue_type="language_mismatch",
                    field="html_lang",
                    severity="medium",
                    message="Page language does not match required language.",
                    suggestion="Fix language attribute or adjust --require-lang.",
                    category="seo.meta",
                    evidence=f"lang={page.html_lang}",
                )
            )
            return [], issues
        return [page], issues

    rf = RequestsFetcher(timeout_seconds=timeout_seconds)
    fr0 = rf.fetch(start_url, allow_redirects=False)
    if fr0.error:
        return (
            [],
            [
                Issue(
                    url=start_url,
                    issue_type="request_failed",
                    field="start_url",
                    severity="high",
                    message="Request failed (timeout/DNS/connection).",
                    suggestion="Check availability and retry.",
                    category="crawl",
                    evidence=fr0.error,
                )
            ],
        )

    if 300 <= fr0.status_code < 400:
        target = normalize_url(urljoin(start_url, fr0.redirect_location)) if fr0.redirect_location else ""
        in_scope = is_in_scope(target, base_domain, scope_prefixes) if target else False
        issues.append(
            Issue(
                url=start_url,
                issue_type="redirect_out_of_scope" if not in_scope else "redirect",
                field="start_url",
                severity="high" if not in_scope else "medium",
                message="Start URL redirects outside configured scope." if not in_scope else "Start URL redirects.",
                suggestion="Fix redirect or adjust --scope-prefix." if not in_scope else "Ensure redirect target is correct.",
                category="crawl",
                evidence=f"status={fr0.status_code} location={target}",
            )
        )
        if not follow_redirects:
            return [], issues

        fr1 = rf.fetch(start_url, allow_redirects=True)
        if fr1.error:
            issues.append(
                Issue(
                    url=start_url,
                    issue_type="request_failed",
                    field="redirect_target",
                    severity="high",
                    message="Failed to follow redirect.",
                    suggestion="Check availability and retry.",
                    category="crawl",
                    evidence=fr1.error,
                )
            )
            return [], issues
        final_url = normalize_url(fr1.final_url)
        if not is_in_scope(final_url, base_domain, scope_prefixes):
            return [], issues
        page = parse_page_data(fr1)
        if require_lang and page.html_lang and not page.html_lang.lower().startswith(require_lang):
            issues.append(
                Issue(
                    url=final_url,
                    issue_type="language_mismatch",
                    field="html_lang",
                    severity="medium",
                    message="Page language does not match required language.",
                    suggestion="Fix language attribute or adjust --require-lang.",
                    category="seo.meta",
                    evidence=f"lang={page.html_lang}",
                )
            )
            return [], issues
        return [page], issues

    final_url = normalize_url(fr0.final_url)
    if not is_in_scope(final_url, base_domain, scope_prefixes):
        issues.append(
            Issue(
                url=start_url,
                issue_type="out_of_scope",
                field="start_url",
                severity="high",
                message="Start URL is not in configured scope.",
                suggestion="Use an in-scope URL or adjust --scope-prefix.",
                category="crawl",
                evidence=f"final={final_url}",
            )
        )
        return [], issues

    page = parse_page_data(fr0)
    if require_lang and page.html_lang and not page.html_lang.lower().startswith(require_lang):
        issues.append(
            Issue(
                url=final_url,
                issue_type="language_mismatch",
                field="html_lang",
                severity="medium",
                message="Page language does not match required language.",
                suggestion="Fix language attribute or adjust --require-lang.",
                category="seo.meta",
                evidence=f"lang={page.html_lang}",
            )
        )
        return [], issues
    return [page], issues


# === ISSUE CHECKS BELOW ===


def local_issues(pages: List[PageData]) -> List[Issue]:
    issues: List[Issue] = []
    title_counts = Counter(p.title for p in pages if p.title)
    desc_counts = Counter(p.meta_description for p in pages if p.meta_description)
    content_counts = Counter(p.content_hash for p in pages if p.content_hash and (p.content_text_len or 0) >= 600)

    def _parse_hsts_max_age(h: str) -> Optional[int]:
        try:
            m = re.search(r"max-age\\s*=\\s*(\\d+)", (h or ""), re.IGNORECASE)
            if not m:
                return None
            return int(m.group(1))
        except Exception:
            return None

    for page in pages:
        url = page.final_url or page.url
        if page.status_code and page.status_code >= 400:
            issues.append(
                Issue(
                    url=url,
                    issue_type="http_error",
                    field="status_code",
                    severity="high",
                    message=f"HTTP status {page.status_code}",
                    suggestion="Fix server/client error or remove broken link.",
                    category="crawl",
                    evidence=str(page.status_code),
                )
            )
            continue
        if page.status_code == 0:
            issues.append(
                Issue(
                    url=url,
                    issue_type="request_failed",
                    field="url",
                    severity="high",
                    message="Request failed (timeout/DNS/connection).",
                    suggestion="Check availability and retry.",
                    category="crawl",
                )
            )
            continue

        if url.startswith("https://"):
            if not (page.hsts or "").strip():
                issues.append(
                    Issue(
                        url=url,
                        issue_type="missing",
                        field="hsts",
                        severity="medium",
                        message="Missing HSTS header (Strict-Transport-Security).",
                        suggestion='Send Strict-Transport-Security (e.g., "max-age=31536000; includeSubDomains").',
                        category="security.headers",
                    )
                )
            else:
                h = (page.hsts or "").strip()
                max_age = _parse_hsts_max_age(h)
                if max_age is not None and max_age < 15552000:
                    issues.append(
                        Issue(
                            url=url,
                            issue_type="weak",
                            field="hsts",
                            severity="low",
                            message="HSTS max-age looks low.",
                            suggestion="Increase max-age (commonly >= 15552000 or 31536000 seconds).",
                            category="security.headers",
                            evidence=h,
                        )
                    )
                if "includesubdomains" not in h.lower():
                    issues.append(
                        Issue(
                            url=url,
                            issue_type="missing_directive",
                            field="hsts",
                            severity="low",
                            message="HSTS header is missing includeSubDomains.",
                            suggestion="Consider adding includeSubDomains (only if all subdomains support HTTPS).",
                            category="security.headers",
                            evidence=h,
                    )
                )

        if page.content_hash and (page.content_text_len or 0) >= 600 and content_counts.get(page.content_hash, 0) > 1:
            issues.append(
                Issue(
                    url=url,
                    issue_type="duplicate",
                    field="content",
                    severity="medium",
                    message="Page content looks duplicated (same text hash on multiple pages).",
                    suggestion="Rewrite/unique-ify main content or add canonical/noindex where appropriate.",
                    category="seo.content",
                    evidence=f"hash={page.content_hash} len={page.content_text_len}",
                )
            )

        if not page.title:
            issues.append(
                Issue(
                    url=url,
                    issue_type="missing",
                    field="title",
                    severity="high",
                    message="Missing <title>.",
                    suggestion="Add a unique, descriptive title (50–60 chars).",
                    category="seo.meta",
                )
            )
        else:
            if len(page.title) < 20:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="length",
                        field="title",
                        severity="medium",
                        message="Title looks too short.",
                        suggestion="Expand title with primary topic + brand.",
                        category="seo.meta",
                        evidence=page.title,
                    )
                )
            if len(page.title) > 70:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="length",
                        field="title",
                        severity="low",
                        message="Title looks too long (may be truncated).",
                        suggestion="Shorten to ~50–60 characters.",
                        category="seo.meta",
                        evidence=page.title,
                    )
                )
            if title_counts.get(page.title, 0) > 1:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="duplicate",
                        field="title",
                        severity="medium",
                        message="Duplicate title across multiple pages.",
                        suggestion="Make titles unique per page intent.",
                        category="seo.meta",
                        evidence=page.title,
                    )
                )

        if not page.meta_description:
            issues.append(
                Issue(
                    url=url,
                    issue_type="missing",
                    field="meta_description",
                    severity="medium",
                    message="Missing meta description.",
                    suggestion="Add a compelling meta description (140–160 chars).",
                    category="seo.meta",
                )
            )
        else:
            if len(page.meta_description) < 70:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="length",
                        field="meta_description",
                        severity="low",
                        message="Meta description looks too short.",
                        suggestion="Expand to ~140–160 characters.",
                        category="seo.meta",
                        evidence=page.meta_description,
                    )
                )
            if len(page.meta_description) > 200:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="length",
                        field="meta_description",
                        severity="low",
                        message="Meta description looks too long.",
                        suggestion="Shorten to ~140–160 characters.",
                        category="seo.meta",
                        evidence=page.meta_description,
                    )
                )
            if desc_counts.get(page.meta_description, 0) > 1:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="duplicate",
                        field="meta_description",
                        severity="low",
                        message="Duplicate meta description across multiple pages.",
                        suggestion="Write unique descriptions per page.",
                        category="seo.meta",
                        evidence=page.meta_description,
                    )
                )

        if not page.h1:
            issues.append(
                Issue(
                    url=url,
                    issue_type="missing",
                    field="h1",
                    severity="low",
                    message="Missing <h1>.",
                    suggestion="Add a clear on-page H1 matching the page topic.",
                    category="seo.headers",
                )
            )

        title_tokens = _tokenize(page.title) | _tokenize(page.meta_description)
        for img in page.images:
            if img.src.startswith(("http://", "https://")) and not is_probable_image_asset_url(img.src):
                parsed = urlparse(img.src)
                sev = "medium" if (parsed.path or "").endswith("/") else "low"
                issues.append(
                    Issue(
                        url=url,
                        issue_type="suspect",
                        field="img_src",
                        severity=sev,
                        message="Image src does not look like a direct image asset (may be broken).",
                        suggestion="Use a direct image file URL (e.g., .webp/.jpg/.png) in the <img> src.",
                        category="images.integrity",
                        evidence=img.src,
                    )
                )
                continue

            if not img.alt:
                issues.append(
                    Issue(
                        url=url,
                        issue_type="missing",
                        field="img_alt",
                        severity="medium",
                        message="Image missing alt text.",
                        suggestion="Add descriptive alt text for accessibility and SEO.",
                        category="images.accessibility",
                        evidence=img.src or img.filename,
                    )
                )
            elif ALT_BAD_REGEX.search(img.alt or ""):
                issues.append(
                    Issue(
                        url=url,
                        issue_type="alt_generic",
                        field="img_alt",
                        severity="high",
                        message="Alt text looks like a placeholder/spam (adsız/şablon/chatgpt/openai vb.).",
                        suggestion="Konuya uygun, anlamlı bir alt metin yazın; adsız/şablon vb. ibareleri kaldırın.",
                        category="images.accessibility",
                        evidence=f"{img.src or img.filename} | alt={img.alt}",
                    )
                )
            elif GENERIC_ALT_RE.match(img.alt):
                issues.append(
                    Issue(
                        url=url,
                        issue_type="generic",
                        field="img_alt",
                        severity="low",
                        message="Alt text looks generic.",
                        suggestion="Use a descriptive alt that matches the image content.",
                        category="images.accessibility",
                        evidence=f"{img.src or img.filename} | alt={img.alt}",
                    )
                )

            if img.alt and re.search(r"\.(png|jpe?g|webp|svg|avif)\b", img.alt, re.IGNORECASE):
                issues.append(
                    Issue(
                        url=url,
                        issue_type="alt_is_filename",
                        field="img_alt",
                        severity="low",
                        message="Alt text looks like a filename (contains an extension).",
                        suggestion="Write alt as a human-readable description, not a file name.",
                        category="images.accessibility",
                        evidence=img.src or img.filename,
                    )
                )

            base = os.path.splitext(img.filename)[0]
            if ALT_BAD_REGEX.search(base or "") or ALT_BAD_REGEX.search(img.src or ""):
                issues.append(
                    Issue(
                        url=url,
                        issue_type="placeholder_src",
                        field="img_src",
                        severity="high",
                        message="Image URL/filename contains placeholder text (adsız/şablon/chatgpt/openai vb.).",
                        suggestion="Dosya adını/URL'yi konuya uygun açıklayıcı kelimelerle güncelleyin.",
                        category="images.accessibility",
                        evidence=img.src or img.filename,
                    )
                )

            if base and (GENERIC_FILENAME_RE.match(base) or CAMERA_FILENAME_RE.match(base)):
                issues.append(
                    Issue(
                        url=url,
                        issue_type="filename",
                        field="img_filename",
                        severity="low",
                        message="Image filename looks non-descriptive.",
                        suggestion="Rename image file to a descriptive, hyphenated name.",
                        category="images.filename",
                        evidence=img.src or img.filename,
                    )
                )

            base_tokens = _tokenize(base)
            alt_tokens = _tokenize(img.alt)
            name_tokens = base_tokens | alt_tokens
            if title_tokens and len(title_tokens) >= 3 and name_tokens and (name_tokens & title_tokens) == set():
                issues.append(
                    Issue(
                        url=url,
                        issue_type="mismatch",
                        field="img_seo",
                        severity="low",
                        message="Image filename/alt does not align with page title/description.",
                        suggestion="Dosya adı veya alt metni sayfa başlığı/konusuyla uyumlu hale getirin.",
                        category="images.seo_meta",
                        evidence=f"{img.filename} | title={page.title}",
                    )
                )

    return issues


def advanced_image_issues(pages: List[PageData], timeout_seconds: int = 20) -> List[Issue]:
    issues: List[Issue] = []

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "from",
        "best",
        "clinic",
        "dental",
        "lema",
        "istanbul",
        "turkey",
        "2023",
        "2024",
        "2025",
    }

    def keywords(text: str) -> Set[str]:
        toks = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {t for t in toks if len(t) >= 3 and t not in stop}

    asset_cache: Dict[str, Dict[str, Any]] = {}

    def head_asset(url: str) -> Dict[str, Any]:
        if url in asset_cache:
            return asset_cache[url]
        headers = {"User-Agent": "real-seo-spider/1.0 (+https://example.local)"}
        out: Dict[str, Any] = {"ok": False, "size": None, "content_type": "", "approx": False, "truncated": False}
        try:
            r = requests.head(url, timeout=timeout_seconds, allow_redirects=True, headers=headers)
            ct = str(r.headers.get("Content-Type") or "")
            cl = r.headers.get("Content-Length")
            size = int(cl) if cl and str(cl).isdigit() else None
            out.update({"ok": True, "size": size, "content_type": ct})
            if size is None:
                # Best-effort fallback: stream-download up to 3MB to estimate size.
                try:
                    resp = requests.get(url, timeout=timeout_seconds, stream=True, headers=headers)
                    resp.raise_for_status()
                    ct2 = str(resp.headers.get("Content-Type") or "")
                    total = 0
                    for chunk in resp.iter_content(chunk_size=65536):
                        if not chunk:
                            break
                        total += len(chunk)
                        if total >= 3_000_000:
                            out["truncated"] = True
                            break
                    out.update({"ok": True, "size": total, "content_type": ct or ct2, "approx": True})
                except Exception:
                    pass
        except Exception:
            pass
        asset_cache[url] = out
        return out

    def jpeg_has_exif(url: str) -> Optional[bool]:
        headers = {"User-Agent": "real-seo-spider/1.0 (+https://example.local)"}
        try:
            resp = requests.get(url, timeout=timeout_seconds, stream=True, headers=headers)
            resp.raise_for_status()
            data = b""
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    break
                data += chunk
                if len(data) >= 512_000:
                    break
            from PIL import Image

            im = Image.open(io.BytesIO(data))
            exif = im.getexif()
            return bool(exif) and len(exif) > 0
        except Exception:
            return None

    def pick_hero(page: PageData) -> Optional[RenderedImage]:
        candidates = [
            im
            for im in page.rendered_images
            if im.in_viewport
            and (im.width * im.height) >= 50_000
            and (im.current_src or im.src)
            and (im.natural_width or 0) > 0
            and (im.natural_height or 0) > 0
        ]
        candidates = [
            im
            for im in candidates
            if (im.current_src or im.src).startswith(("http://", "https://"))
            and not (im.current_src or im.src).startswith("data:")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda im: im.width * im.height)

    hero_by_page: Dict[str, str] = {}
    for page in pages:
        hero = pick_hero(page)
        if hero is not None:
            hero_by_page[page.final_url] = hero.current_src or hero.src

    hero_counts = Counter(v for v in hero_by_page.values() if v)
    for page in pages:
        hero_url = hero_by_page.get(page.final_url, "")
        if hero_url and hero_counts.get(hero_url, 0) > 1:
            issues.append(
                Issue(
                    url=page.final_url,
                    issue_type="duplicate",
                    field="img_hero",
                    severity="medium",
                    message="Same hero image appears on multiple pages (duplicate hero).",
                    suggestion="Use unique hero visuals to improve uniqueness and intent match.",
                    category="images.relevance",
                    evidence=hero_url,
                )
            )

    for page in pages:
        page_url = page.final_url or page.url
        hero = pick_hero(page)
        if hero is not None:
            hero_url = hero.current_src or hero.src
            if hero.loading.strip().lower() == "lazy":
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="loading",
                        field="img_loading",
                        severity="high",
                        message="Above-the-fold hero image is lazy-loaded (can hurt LCP).",
                        suggestion='Remove lazy-load for LCP image; consider fetchpriority="high" and/or preload.',
                        category="images.loading",
                        evidence=hero_url,
                    )
                )

            preload_bases = {_strip_query_fragment(u) for u in (page.preload_images or ()) if u}
            hero_base = _strip_query_fragment(hero_url)
            has_preload = bool(hero_base and hero_base in preload_bases)
            has_high_priority = hero.fetchpriority.strip().lower() == "high"
            if hero_url and not has_preload and not has_high_priority:
                meta = head_asset(hero_url)
                size = meta.get("size")
                sev = "medium" if isinstance(size, int) and size >= 200_000 else "low"
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="priority",
                        field="img_lcp_priority",
                        severity=sev,
                        message="Above-the-fold hero image is not prioritized (no preload / fetchpriority=high).",
                        suggestion='For the LCP/hero image, consider adding fetchpriority="high" and/or a <link rel="preload" as="image">.',
                        category="images.performance",
                        evidence=hero_url,
                    )
                )

            intent = keywords(f"{page.title} {page.h1}")
            fname = urlparse(hero_url).path.split("/")[-1]
            hero_tokens = keywords(f"{fname} {hero.alt}")
            if intent and not (intent & hero_tokens) and (GENERIC_ALT_RE.match(hero.alt) or "hero" in fname.lower()):
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="relevance_low_confidence",
                        field="img_relevance",
                        severity="low",
                        message="Hero image may be weakly related to page intent (low-confidence heuristic).",
                        suggestion="Use a hero image that matches the page's main intent (H1/title) and name it descriptively.",
                        category="images.relevance",
                        evidence=hero_url,
                    )
                )

        rendered = list(page.rendered_images)
        if not rendered:
            continue

        seen_path_flags: Set[str] = set()

        for im in rendered:
            img_url = im.current_src or im.src
            if not img_url.startswith(("http://", "https://")):
                continue
            if "/blank.gif" in img_url.lower():
                continue
            area = im.width * im.height
            if area < 20_000:
                continue

            path = (urlparse(img_url).path or "").lower()
            if "/wp-content/uploads/" in path and re.search(r"/20\\d{2}/\\d{2}/", path) and img_url not in seen_path_flags:
                seen_path_flags.add(img_url)
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="path",
                        field="img_path",
                        severity="low",
                        message="Image is served from a date-based uploads path (hard to keep assets categorized).",
                        suggestion="Prefer a categorized folder structure for evergreen assets (e.g., /images/treatments/...).",
                        category="images.path",
                        evidence=img_url,
                    )
                )

            if not im.in_viewport and im.loading.strip().lower() not in ("lazy",):
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="loading",
                        field="img_loading",
                        severity="low",
                        message="Below-the-fold image is not lazy-loaded.",
                        suggestion='Add loading="lazy" and decoding="async" for below-the-fold images.',
                        category="images.loading",
                        evidence=img_url,
                    )
                )

            if not im.in_viewport and im.decoding.strip().lower() != "async":
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="decoding",
                        field="img_decoding",
                        severity="low",
                        message='Below-the-fold image is missing decoding="async".',
                        suggestion='Add decoding="async" (and loading="lazy") for below-the-fold images.',
                        category="images.loading",
                        evidence=img_url,
                    )
                )

            if im.width >= 650 and not im.srcset and not im.sizes:
                issues.append(
                    Issue(
                        url=page_url,
                        issue_type="responsive",
                        field="img_responsive",
                        severity="medium",
                        message="Large image is missing srcset/sizes (responsive).",
                        suggestion="Provide srcset + sizes so browsers can choose the right size.",
                        category="images.responsive",
                        evidence=img_url,
                    )
                )

            if (im.natural_width or 0) <= 0 or (im.natural_height or 0) <= 0:
                continue

            meta = head_asset(img_url)
            size = meta.get("size")
            ct = str(meta.get("content_type") or "").lower()
            if isinstance(size, int) and size > 0:
                kb = int(size / 1024)
                if size >= 1_000_000:
                    sev = "high"
                elif size >= 300_000:
                    sev = "medium" if im.in_viewport else "low"
                else:
                    sev = ""
                if sev:
                    issues.append(
                        Issue(
                            url=page_url,
                            issue_type="size",
                            field="img_size",
                            severity=sev,
                            message=f"Image is large (~{kb} KB).",
                            suggestion="Compress/resize; aim for LCP image <200–300KB when possible.",
                            category="images.performance",
                            evidence=img_url,
                        )
                    )

            ext = os.path.splitext(urlparse(img_url).path)[1].lower()
            if ("image/jpeg" in ct or "image/png" in ct or ext in (".jpg", ".jpeg", ".png")) and isinstance(size, int):
                if size >= 200_000:
                    issues.append(
                        Issue(
                            url=page_url,
                            issue_type="format",
                            field="img_format",
                            severity="low",
                            message="Large JPG/PNG detected; modern formats may improve performance.",
                            suggestion="Use WebP/AVIF for photos; SVG for icons (with fallback if needed).",
                            category="images.performance",
                            evidence=img_url,
                        )
                    )

            if ("image/jpeg" in ct or ext in (".jpg", ".jpeg")):
                has_exif = jpeg_has_exif(img_url)
                if has_exif is True:
                    issues.append(
                        Issue(
                            url=page_url,
                            issue_type="exif",
                            field="img_exif",
                            severity="low",
                            message="JPEG contains EXIF metadata (privacy/performance).",
                            suggestion="Strip EXIF metadata unless explicitly needed.",
                            category="images.privacy",
                            evidence=img_url,
                        )
                    )

    return issues


def gpt_issues(pages: List[PageData], api_key: str, model: str, sleep_seconds: float = 0.0) -> List[Issue]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    out: List[Issue] = []

    system = (
        "Sen deneyimli bir SEO uzmanısın. Çıktıyı TÜRKÇE yaz. "
        "Sadece verilen veri üzerinden konuş; uydurma yapma. "
        "Title kalitesi, meta description kalitesi, title↔meta description uyumu, H1 uyumu ve görsel alt metinleri "
        "üzerinden denetim yap. Sadece JSON döndür."
    )

    for page in pages:
        url = page.final_url or page.url
        images_payload = [{"filename": i.filename, "alt": i.alt, "src": i.src} for i in page.images]
        payload = {
            "url": url,
            "title": page.title,
            "meta_description": page.meta_description,
            "h1": page.h1,
            "canonical": page.canonical,
            "images": images_payload[:25],
        }

        user = (
            "Audit this page data for SEO suitability.\n"
            "If something is unsuitable, add an issue.\n"
            "Do NOT invent page content beyond provided fields.\n\n"
            "Ek zorunlu kontrol:\n"
            "- Title ve meta description aynı konu/intent'i anlatmalı. Uyuşmazlık varsa issue ekle:\n"
            '  issue_type="mismatch", field="title_meta_desc", severity="medium".\n\n'
            "Return STRICT JSON object with this shape:\n"
            '{ \"issues\": [ { \"issue_type\": \"...\", \"field\": \"...\", \"severity\": \"low|medium|high\", '
            '\"message\": \"...\", \"suggestion\": \"...\", \"evidence\": \"...\" } ] }\n\n'
            f"PAGE_DATA:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw) if raw else {"issues": []}
        except Exception as e:
            out.append(
                Issue(
                    url=url,
                    issue_type="gpt_error",
                    field="gpt",
                    severity="low",
                    message="GPT audit failed for this page.",
                    suggestion="Retry or check your API key / rate limits.",
                    category="gpt",
                    evidence=str(e),
                    source="gpt",
                )
            )
            continue

        for it in data.get("issues", []):
            field = str(it.get("field", "")).strip() or "unspecified"
            issue_type = str(it.get("issue_type", "")).strip() or "unspecified"
            category = "gpt"
            if field == "title_meta_desc" and issue_type in ("mismatch", "misaligned", "not_aligned"):
                category = "seo.meta"
            out.append(
                Issue(
                    url=url,
                    issue_type=issue_type,
                    field=field,
                    severity=str(it.get("severity", "low")).strip(),
                    message=str(it.get("message", "")).strip(),
                    suggestion=str(it.get("suggestion", "")).strip(),
                    category=category,
                    evidence=str(it.get("evidence", "")).strip(),
                    source="gpt",
                )
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return out


# === OUTPUT / CLI BELOW ===


def compute_seo_score(pages: List[PageData], issues: List[Issue]) -> int:
    pages_n = max(1, len(pages))
    weights = {"high": 12, "medium": 6, "low": 2}
    total = 0
    for it in issues:
        total += int(weights.get((it.severity or "").lower(), 1))
    # Normalize to pages to avoid huge crawls always scoring 0.
    per_page = total / pages_n
    score = int(round(max(0.0, 100.0 - per_page)))
    return max(0, min(100, score))


def gpt_summaries(
    pages: List[PageData],
    issues: List[Issue],
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    issues_by_url: Dict[str, List[Issue]] = {}
    for it in issues:
        issues_by_url.setdefault(it.url, []).append(it)

    # Keep payload compact (avoid huge prompts on big crawls).
    def compact_issue(i: Issue) -> Dict[str, Any]:
        ev = (i.evidence or "").strip()
        if ev.startswith("data:") or ev.startswith("data://"):
            ev = f"inline-data ({len(ev)} chars)"
        if len(ev) > 220:
            ev = ev[:220] + "..."
        return {
            "severity": i.severity,
            "category": i.category,
            "field": i.field,
            "issue_type": i.issue_type,
            "message": i.message,
            "evidence": ev,
        }

    pages_payload: List[Dict[str, Any]] = []
    for p in pages[:60]:
        url = p.final_url or p.url
        page_issues = sorted(
            issues_by_url.get(url, []),
            key=lambda x: ({"high": 0, "medium": 1, "low": 2}.get(x.severity, 9), x.category, x.field, x.issue_type),
        )[:18]
        images_payload: List[Dict[str, str]] = []
        for im in p.images[:40]:
            src = (im.src or "").strip()
            if not src:
                continue
            if src.startswith(("data:", "data://")):
                continue
            images_payload.append({"src": src, "filename": im.filename, "alt": im.alt})
        pages_payload.append(
            {
                "url": url,
                "title": p.title,
                "h1": p.h1,
                "meta_description": p.meta_description,
                "issues": [compact_issue(i) for i in page_issues],
                "images": images_payload,
            }
        )

    system = (
        "Sen deneyimli bir SEO ve web performans uzmanısın. "
        "Çıktıyı TÜRKÇE yaz. "
        "Kullanıcıya daha kapsamlı, net ve aksiyon odaklı özet ver. "
        "Sadece verilen veri üzerinden konuş; uydurma yapma. "
        "Title/meta/H1 uyumu, hreflang/html lang, görsel alt/format/lazy-load, performans ve güvenlik başlıklarını da değerlendir."
    )

    user = (
        "Aşağıdaki tarama verisi için:\n"
        "1) Site/origin için daha detaylı genel özet çıkar:\n"
        "   - 8-14 cümle, aksiyon odaklı, tek paragraf.\n"
        "   - 4-6 maddelik 'Öncelikli aksiyonlar' listesi ekle.\n"
        "   - 2-3 tane 'Örnek' ver: (ör. belirli sayfa URL'lerinde title/meta uyumsuzluğu, alt eksikliği, LCP görseli).\n"
        "2) Her sayfa için 1-2 cümlelik 'Bu sayfanın genel sorunu' özetini yaz.\n\n"
        "3) Her sayfa için görsellerin sayfanın Title + Meta Description ile ne kadar alakalı olduğunu 0-100 arasında puanla.\n"
        "   Sadece alt + filename + src üzerinden değerlendir; görselin piksel içeriğini bilmediğini varsay.\n\n"
        "ÇIKTIYI SADECE JSON ver.\n"
        "Şema:\n"
        '{ "site_summary": "....", "page_summaries": { "url": "summary", "...": "..." }, '
        '"image_relevance": { "url": { "image_src": 0, "image_src2": 0 } } }\n\n'
        f"DATA:\n{json.dumps({'pages': pages_payload}, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = json.loads(raw) if raw else {}

    site_summary = str(data.get("site_summary") or "").strip()
    page_summaries = data.get("page_summaries") or {}
    if not isinstance(page_summaries, dict):
        page_summaries = {}

    image_relevance_raw = data.get("image_relevance") or {}
    if not isinstance(image_relevance_raw, dict):
        image_relevance_raw = {}

    # Safety clamp for UI/persistence.
    if len(site_summary) > 3200:
        site_summary = site_summary[:3200] + "..."
    cleaned_page_summaries: Dict[str, str] = {}
    for k, v in page_summaries.items():
        key = str(k or "").strip()
        val = str(v or "").strip()
        if not key or not val:
            continue
        if len(val) > 650:
            val = val[:650] + "..."
        cleaned_page_summaries[key] = val

    cleaned_image_relevance: Dict[str, Dict[str, int]] = {}
    for page_url, mapping in image_relevance_raw.items():
        purl = str(page_url or "").strip()
        if not purl or not isinstance(mapping, dict):
            continue
        out_map: Dict[str, int] = {}
        for src, score in mapping.items():
            s = str(src or "").strip()
            if not s:
                continue
            try:
                n = int(score)
            except Exception:
                continue
            n = max(0, min(100, n))
            out_map[s] = n
        if out_map:
            cleaned_image_relevance[purl] = out_map

    return {
        "site_summary": site_summary,
        "page_summaries": cleaned_page_summaries,
        "image_relevance": cleaned_image_relevance,
    }


def dedupe_issues(items: List[Issue]) -> List[Issue]:
    seen: Set[Tuple[str, str, str, str, str]] = set()
    out: List[Issue] = []
    for it in items:
        key = (it.url, it.category, it.issue_type, it.field, it.evidence)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def write_csv(path: str, issues: List[Issue]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url",
                "category",
                "issue_type",
                "field",
                "severity",
                "message",
                "suggestion",
                "evidence",
                "source",
            ],
        )
        writer.writeheader()
        for i in issues:
            writer.writerow(dataclasses.asdict(i))


def write_json(path: str, pages: List[PageData], issues: List[Issue], extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "pages": [dataclasses.asdict(p) for p in pages],
        "issues": [dataclasses.asdict(i) for i in issues],
        "stats": {"pages": len(pages), "issues": len(issues)},
    }
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crawl a domain (optionally with Chromium) and audit SEO + images, exporting CSV/JSON."
    )
    parser.add_argument("--start", default=_get_cfg("START_URL", "https://lemaclinic.com/"))
    parser.add_argument("--domain", default=_get_cfg("BASE_DOMAIN", "lemaclinic.com"))
    parser.add_argument(
        "--scope-prefix",
        action="append",
        default=None,
        help="In-scope URL path prefix (repeatable). Examples: --scope-prefix /  or  --scope-prefix /en/",
    )
    parser.add_argument(
        "--require-lang",
        default=_get_cfg("REQUIRE_LANG", ""),
        help='Optional: require <html lang> to start with this value (example: \"en\").',
    )
    parser.add_argument("--max-pages", type=int, default=int(_get_cfg("MAX_PAGES", 200)))
    parser.add_argument("--timeout", type=int, default=int(_get_cfg("REQUEST_TIMEOUT_SECONDS", 30)))
    parser.add_argument("--out-csv", default="out/issues.csv")
    parser.add_argument("--out-json", default="out/report.json")
    parser.add_argument("--no-gpt", action="store_true", help="Skip GPT analysis (local checks only).")
    parser.add_argument("--fetcher", choices=["requests", "chromium"], default="requests")
    parser.add_argument("--headed", action="store_true", help="Use visible Chromium window (debug).")
    parser.add_argument(
        "--mode",
        choices=["single", "crawl"],
        default="single",
        help="single: only --start, crawl: discover more pages.",
    )
    parser.add_argument("--follow-redirects", action="store_true")
    parser.add_argument("--use-sitemap", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between GPT calls (seconds).")
    args = parser.parse_args()

    scope_prefixes = args.scope_prefix if args.scope_prefix else _get_list_cfg("SCOPE_PREFIXES", ["/"])
    scope_prefixes = [_normalize_prefix(p) for p in scope_prefixes]

    if args.mode == "single":
        pages, seed_issues = audit_single(
            start_url=args.start,
            base_domain=args.domain,
            scope_prefixes=scope_prefixes,
            timeout_seconds=args.timeout,
            follow_redirects=args.follow_redirects,
            fetcher=args.fetcher,
            chromium_headless=(not args.headed),
            require_lang=args.require_lang,
        )
    else:
        pages, seed_issues = crawl(
            start_url=args.start,
            base_domain=args.domain,
            scope_prefixes=scope_prefixes,
            max_pages=args.max_pages,
            timeout_seconds=args.timeout,
            use_sitemap=args.use_sitemap,
            fetcher=args.fetcher,
            chromium_headless=(not args.headed),
            require_lang=args.require_lang,
        )

    issues: List[Issue] = []
    issues.extend(seed_issues)
    issues.extend(local_issues(pages))
    issues.extend(advanced_image_issues(pages, timeout_seconds=min(30, args.timeout)))

    api_key = str(_get_cfg("OPENAI_API_KEY", "")).strip()
    model = str(_get_cfg("OPENAI_MODEL", "gpt-4o-mini")).strip()
    if not args.no_gpt and api_key:
        issues.extend(gpt_issues(pages, api_key=api_key, model=model, sleep_seconds=args.sleep))

    issues = dedupe_issues(issues)

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (severity_rank.get(i.severity, 9), i.url, i.category, i.field, i.issue_type))

    write_csv(args.out_csv, issues)
    write_json(args.out_json, pages, issues)

    print(f"Crawled {len(pages)} pages.")
    print(f"Wrote {len(issues)} issues to {args.out_csv}")
    print(f"Wrote full report to {args.out_json}")
    if not api_key and not args.no_gpt:
        print("Note: OPENAI_API_KEY not set; GPT audit was skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
