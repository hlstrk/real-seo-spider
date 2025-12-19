from __future__ import annotations

import dataclasses
import json
import os
import runpy
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import seo_audit


@dataclass
class Job:
    id: str
    created_at: float
    status: str  # queued|running|done|error
    error: str = ""
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    pages: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    issues: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    seo_score: int = 0
    site_summary: str = ""
    page_summaries: Dict[str, str] = dataclasses.field(default_factory=dict)
    image_relevance: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)
    out_dir: str = ""


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUT_DIR = Path("out") / "web"
JOBS_DB_PATH = OUT_DIR / "jobs.json"
SHOTS_DIRNAME = "shots"

# Auth (very simple, local use). Change these if needed.
ADMIN_USERNAME = "lema"
ADMIN_PASSWORD = "1234"

# OpenAI API key: do NOT ask for it in the UI. Paste it here or set env var OPENAI_API_KEY.
# Prefer env var for safety.
OPENAI_API_KEY_OVERRIDE = ""

app = FastAPI(title="real-seo-spider web")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

TR: Dict[str, Any] = {
    "status": {"queued": "sırada", "running": "çalışıyor", "done": "tamamlandı", "error": "hata"},
    "severity": {"high": "yüksek", "medium": "orta", "low": "düşük"},
    "issue_type": {
        "missing": "eksik",
        "duplicate": "kopya",
        "length": "uzunluk",
        "generic": "genel/kalıplaşmış",
        "alt_is_filename": "alt dosya adı gibi",
        "filename": "dosya adı",
        "responsive": "responsive eksik",
        "loading": "yükleme",
        "size": "boyut",
        "format": "format",
        "exif": "exif",
        "request_failed": "istek başarısız",
        "http_error": "http hatası",
        "redirect_out_of_scope": "kapsam dışına yönlendirme",
        "language_mismatch": "dil uyuşmuyor",
        "mismatch": "uyumsuz",
        "priority": "öncelik",
        "path": "path",
        "decoding": "decoding",
        "suspect": "şüpheli",
        "weak": "zayıf",
        "missing_directive": "directive eksik",
    },
    "field": {
        "title": "title",
        "meta_description": "meta açıklama",
        "h1": "h1",
        "canonical": "canonical",
        "html_lang": "html lang",
        "status_code": "durum kodu",
        "start_url": "başlangıç url",
        "img_alt": "görsel alt",
        "img_filename": "görsel dosya adı",
        "img_src": "görsel kaynak",
        "img_hero": "hero görsel",
        "img_loading": "görsel lazy-load",
        "img_responsive": "görsel srcset/sizes",
        "img_size": "görsel boyutu",
        "img_format": "görsel formatı",
        "img_exif": "görsel exif",
        "img_lcp_priority": "lcp önceliği",
        "img_path": "görsel path",
        "img_decoding": "görsel decoding",
        "title_meta_desc": "title ↔ meta açıklama",
        "hsts": "hsts",
        "content": "içerik",
        "gpt": "gpt",
    },
}
templates.env.globals["TR"] = TR

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("WEBAPP_SESSION_SECRET", "dev-secret-change-me"),
    same_site="lax",
)


def _load_openai_key_from_files() -> str:
    # Priority:
    # 1) OPENAI_API_KEY_OVERRIDE in this file
    # 2) env var OPENAI_API_KEY
    # 3) config.py (recommended)
    # 4) config.example.py (fallback, user requested)
    if OPENAI_API_KEY_OVERRIDE.strip():
        return OPENAI_API_KEY_OVERRIDE.strip()

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        import config  # type: ignore

        key = str(getattr(config, "OPENAI_API_KEY", "") or "").strip()
        if key:
            return key
    except Exception:
        pass

    try:
        cfg = runpy.run_path(str(Path("config.example.py").resolve()))
        key = str(cfg.get("OPENAI_API_KEY", "") or "").strip()
        return key
    except Exception:
        return ""


def _load_openai_model_from_files() -> str:
    env_model = os.getenv("OPENAI_MODEL", "").strip()
    if env_model:
        return env_model
    try:
        import config  # type: ignore

        m = str(getattr(config, "OPENAI_MODEL", "") or "").strip()
        if m:
            return m
    except Exception:
        pass
    try:
        cfg = runpy.run_path(str(Path("config.example.py").resolve()))
        m = str(cfg.get("OPENAI_MODEL", "") or "").strip()
        return m
    except Exception:
        return ""


def _is_logged_in(request: Request) -> bool:
    return bool(request.session.get("auth") == "ok")


def _require_login(request: Request) -> None:
    if not _is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")


def _load_jobs_db() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_DB_PATH.exists():
        return
    try:
        data = json.loads(JOBS_DB_PATH.read_text(encoding="utf-8"))
        items = data.get("jobs", [])
        with JOBS_LOCK:
            for it in items:
                job = Job(
                    id=str(it["id"]),
                    created_at=float(it.get("created_at") or time.time()),
                    status=str(it.get("status") or "done"),
                    error=str(it.get("error") or ""),
                    params=dict(it.get("params") or {}),
                    pages=list(it.get("pages") or []),
                    issues=list(it.get("issues") or []),
                    seo_score=int(it.get("seo_score") or 0),
                    site_summary=str(it.get("site_summary") or ""),
                    page_summaries=dict(it.get("page_summaries") or {}),
                    image_relevance=dict(it.get("image_relevance") or {}),
                    out_dir=str(it.get("out_dir") or ""),
                )
                JOBS[job.id] = job
    except Exception:
        # If corrupt, ignore. (Don't crash web UI)
        return


def _save_jobs_db() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        jobs_sorted = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)[:200]
        payload = {"jobs": [dataclasses.asdict(j) for j in jobs_sorted]}
    JOBS_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.status = "running"

    try:
        p = job.params
        start_url = str(p["start_url"])
        domain = str(p["domain"])
        mode = str(p["mode"])
        fetcher = str(p["fetcher"])
        scope_prefixes = [s.strip() for s in str(p["scope_prefixes"]).split(",") if s.strip()]
        require_lang = str(p.get("require_lang") or "")
        max_pages = int(p["max_pages"])
        timeout = int(p["timeout"])
        use_sitemap = bool(p.get("use_sitemap"))
        follow_redirects = bool(p.get("follow_redirects"))
        no_gpt = bool(p.get("no_gpt"))
        sleep = float(p.get("sleep") or 0)
        model = str(p.get("model") or (_load_openai_model_from_files() or "gpt-4o-mini"))
        api_key = _load_openai_key_from_files()

        # Default to chromium for Cloudflare-protected pages.
        if fetcher not in ("chromium", "requests"):
            fetcher = "chromium"

        # Run audit
        if mode == "single":
            pages, seed_issues = seo_audit.audit_single(
                start_url=start_url,
                base_domain=domain,
                scope_prefixes=scope_prefixes,
                timeout_seconds=timeout,
                follow_redirects=follow_redirects,
                fetcher=fetcher,
                chromium_headless=True,
                require_lang=require_lang,
            )
        else:
            pages, seed_issues = seo_audit.crawl(
                start_url=start_url,
                base_domain=domain,
                scope_prefixes=scope_prefixes,
                max_pages=max_pages,
                timeout_seconds=timeout,
                use_sitemap=use_sitemap,
                fetcher=fetcher,
                chromium_headless=True,
                require_lang=require_lang,
            )

        issues: List[seo_audit.Issue] = []
        issues.extend(seed_issues)
        issues.extend(seo_audit.local_issues(pages))
        issues.extend(seo_audit.advanced_image_issues(pages, timeout_seconds=min(30, timeout)))

        site_summary = ""
        page_summaries: Dict[str, str] = {}
        image_relevance: Dict[str, Dict[str, int]] = {}
        if not no_gpt and api_key:
            issues.extend(seo_audit.gpt_issues(pages, api_key=api_key, model=model, sleep_seconds=sleep))
            try:
                summ = seo_audit.gpt_summaries(pages, issues, api_key=api_key, model=model)
                site_summary = str(summ.get("site_summary") or "")
                page_summaries = dict(summ.get("page_summaries") or {})
                image_relevance = dict(summ.get("image_relevance") or {})
            except Exception:
                site_summary = ""
                page_summaries = {}
                image_relevance = {}

        issues = seo_audit.dedupe_issues(issues)
        seo_score = seo_audit.compute_seo_score(pages, issues)

        severity_rank = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda i: (severity_rank.get(i.severity, 9), i.url, i.category, i.field, i.issue_type))

        pages_dict = [dataclasses.asdict(p) for p in pages]
        issues_dict = [dataclasses.asdict(i) for i in issues]

        out_dir = OUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        seo_audit.write_json(
            str(out_dir / "report.json"),
            pages,
            issues,
            extra={
                "seo_score": seo_score,
                "site_summary": site_summary,
                "page_summaries": page_summaries,
                "image_relevance": image_relevance,
            },
        )
        seo_audit.write_csv(str(out_dir / "issues.csv"), issues)

        with JOBS_LOCK:
            job.pages = pages_dict
            job.issues = issues_dict
            job.seo_score = seo_score
            job.site_summary = site_summary
            job.page_summaries = page_summaries
            job.image_relevance = image_relevance
            job.out_dir = str(out_dir)
            job.status = "done"
        _save_jobs_db()
    except Exception as e:
        with JOBS_LOCK:
            job = JOBS[job_id]
            job.status = "error"
            job.error = str(e)
        _save_jobs_db()


def _jobs_list() -> List[Job]:
    with JOBS_LOCK:
        return sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)


_load_jobs_db()


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request) -> HTMLResponse:
    if _is_logged_in(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/login")
def login_post(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        request.session["auth"] = "ok"
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Hatalı kullanıcı adı veya şifre."})


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    if not _is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "jobs": _jobs_list()[:25],
            "defaults": {
                "start_url": "https://lemaclinic.com/",
                "domain": "lemaclinic.com",
                "mode": "crawl",
                "fetcher": "chromium",
                "scope_prefixes": "/",
                "require_lang": "en",
                "max_pages": "50",
                "timeout": "30",
                "use_sitemap": False,
                "follow_redirects": False,
                "no_gpt": False,
                "sleep": "0",
                "model": (_load_openai_model_from_files() or "gpt-4o-mini"),
            },
        },
    )


@app.post("/scan")
def start_scan(
    request: Request,
    start_url: str = Form(...),
    domain: str = Form(...),
    mode: str = Form("crawl"),
    fetcher: str = Form("chromium"),
    scope_prefixes: str = Form("/"),
    require_lang: str = Form(""),
    max_pages: int = Form(200),
    timeout: int = Form(30),
    use_sitemap: bool = Form(False),
    follow_redirects: bool = Form(False),
    no_gpt: bool = Form(False),
    sleep: float = Form(0),
    model: str = Form("gpt-4o-mini"),
) -> RedirectResponse:
    if not _is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    job_id = _new_job_id()
    out_dir = str(OUT_DIR / job_id)
    job = Job(
        id=job_id,
        created_at=time.time(),
        status="queued",
        params={
            "start_url": start_url,
            "domain": domain,
            "mode": mode,
            "fetcher": fetcher,
            "scope_prefixes": scope_prefixes,
            "require_lang": require_lang,
            "max_pages": max_pages,
            "timeout": timeout,
            "use_sitemap": use_sitemap,
            "follow_redirects": follow_redirects,
            "no_gpt": no_gpt,
            "sleep": sleep,
            "model": model,
        },
        out_dir=out_dir,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    _save_jobs_db()

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return RedirectResponse(url=f"/results/{job_id}", status_code=303)


@app.get("/results/{job_id}", response_class=HTMLResponse)
def results(request: Request, job_id: str) -> HTMLResponse:
    if not _is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "job": None, "job_id": job_id, "refresh": False},
            status_code=404,
        )

    refresh = job.status in ("queued", "running")

    category_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for it in job.issues:
        cat = str(it.get("category") or "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        sev = str(it.get("severity") or "").lower()
        if sev in severity_counts:
            severity_counts[sev] += 1
    categories = sorted(category_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def score_for(prefixes: List[str]) -> int:
        pages_n = max(1, len(job.pages))
        weights = {"high": 12, "medium": 6, "low": 2}
        total = 0
        for it in job.issues:
            cat = str(it.get("category") or "")
            if not any(cat.startswith(p) for p in prefixes):
                continue
            sev = str(it.get("severity") or "").lower()
            total += int(weights.get(sev, 1))
        per_page = total / pages_n
        return max(0, min(100, int(round(100 - per_page))))

    donuts = {
        "meta": score_for(["seo.meta"]),
        "headers": score_for(["seo.headers"]),
        "images": score_for(["images."]),
        "performance": score_for(["images.performance", "images.loading", "images.responsive"]),
    }

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "job": job,
            "job_id": job_id,
            "refresh": refresh,
            "pages_count": len(job.pages),
            "issues_count": len(job.issues),
            "jobs": _jobs_list()[:25],
            "categories": categories,
            "severity_counts": severity_counts,
            "seo_score": int(getattr(job, "seo_score", 0) or 0),
            "site_summary": str(getattr(job, "site_summary", "") or ""),
            "page_summaries": dict(getattr(job, "page_summaries", {}) or {}),
            "image_relevance": dict(getattr(job, "image_relevance", {}) or {}),
            "donuts": donuts,
        },
    )


def _shot_path(job_id: str, idx: int) -> Path:
    return OUT_DIR / job_id / SHOTS_DIRNAME / f"{idx}.png"


def _ensure_issue_screenshot(job: Job, idx: int) -> Path:
    out_path = _shot_path(job.id, idx)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    issue = job.issues[idx]
    page_url = str(issue.get("url") or "")
    timeout_ms = 45_000

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependencies for screenshots. Install: pip install opencv-python numpy playwright") from e

    find_target_js = r"""
      (issue) => {
        const norm = (s) => (s || '').trim();
        const evidence = norm(issue?.evidence || '');
        const issueType = norm(issue?.issue_type || '');
        const field = norm(issue?.field || '');

        const isImg = field.startsWith('img_');
        const imgs = Array.from(document.querySelectorAll('img'));

        function stripQ(u) {
          return (u || '').split('#')[0].split('?')[0].trim();
        }
        function looksLikeImgAsset(u) {
          const b = stripQ(u).toLowerCase();
          return b.startsWith('data:') || /\.(avif|webp|png|jpe?g|gif|svg|ico|bmp)$/.test(b);
        }
        function isLoaded(el) {
          return !!(el && el.complete && (el.naturalWidth || 0) > 0);
        }

        function candidatesFor(el) {
          const cur = norm(el.currentSrc || el.src || '');
          const src = norm(el.getAttribute('src') || '');
          const ds = norm(el.getAttribute('data-src') || '');
          const dls = norm(el.getAttribute('data-lazy-src') || '');
          const dor = norm(el.getAttribute('data-original') || '');
          return [cur, ds, dls, dor, src].filter(Boolean);
        }

        function bestMatchByEvidence() {
          if (!evidence || evidence.startsWith('data:') || evidence.startsWith('data://')) return null;
          let bestEl = null;
          let bestScore = 0;
          for (const el of imgs) {
            const pool = candidatesFor(el);
            let score = 0;
            for (const u of pool) {
              if (u === evidence) score = Math.max(score, 3);
              else if (stripQ(u) === stripQ(evidence)) score = Math.max(score, 2);
              else if (u.includes(evidence) || evidence.includes(u)) score = Math.max(score, 1);
            }
            if (score > bestScore) { bestScore = score; bestEl = el; }
          }
          return bestScore > 0 ? bestEl : null;
        }

        function firstMissingAlt() {
          return imgs.find((el) => {
            if (norm(el.getAttribute('alt'))) return false;
            const u = norm(el.currentSrc || el.src || '');
            if (!looksLikeImgAsset(u)) return false;
            return true;
          });
        }

        function firstLargeNoSrcset() {
          return imgs.find((el) => {
            const r = el.getBoundingClientRect();
            const w = r.width || 0;
            const h = r.height || 0;
            if (w < 650 || h < 250) return false;
            const srcset = norm(el.getAttribute('srcset'));
            const sizes = norm(el.getAttribute('sizes'));
            return !srcset && !sizes;
          });
        }

        function firstBelowFoldNotLazy() {
          const vh = window.innerHeight || 900;
          return imgs.find((el) => {
            const r = el.getBoundingClientRect();
            const below = (r.top || 0) > vh;
            if (!below) return false;
            const loading = norm(el.getAttribute('loading')).toLowerCase();
            return loading !== 'lazy';
          });
        }

        let el = null;
        if (isImg) {
          el = bestMatchByEvidence();
          if (!el && issueType === 'missing' && field === 'img_alt') el = firstMissingAlt();
          if (!el && issueType === 'responsive' && field === 'img_responsive') el = firstLargeNoSrcset();
          if (!el && issueType === 'loading' && field === 'img_loading') el = firstBelowFoldNotLazy();
        }

        if (!el) return null;
        el.scrollIntoView({ block: 'center', inline: 'center' });
        const idx = imgs.indexOf(el);
        const r = el.getBoundingClientRect();
        return {
          idx,
          rect: { x: r.left || 0, y: r.top || 0, w: r.width || 0, h: r.height || 0 },
          loaded: isLoaded(el),
          url: norm(el.currentSrc || el.src || '')
        };
      }
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            locale="en-US",
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.goto(page_url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        box = None
        target_idx = None
        try:
            res = page.evaluate(find_target_js, issue)
            if isinstance(res, dict):
                target_idx = res.get("idx")
        except Exception:
            target_idx = None

        if isinstance(target_idx, int) and target_idx >= 0:
            try:
                loc = page.locator("img").nth(target_idx)
                loc.scroll_into_view_if_needed(timeout=5_000)
                try:
                    page.wait_for_function(
                        "(i) => { const el = document.querySelectorAll('img')[i]; return !!(el && el.complete && (el.naturalWidth||0)>0); }",
                        target_idx,
                        timeout=5_000,
                    )
                except Exception:
                    pass
                try:
                    loc.evaluate(
                        "el => { el.style.outline = '4px solid rgba(255,45,85,1)'; el.style.outlineOffset = '3px'; el.style.borderRadius = '10px'; }"
                    )
                except Exception:
                    pass

                bbox = loc.bounding_box()
                scroll = page.evaluate(
                    "() => ({ sx: window.scrollX||0, sy: window.scrollY||0, dpr: window.devicePixelRatio||1 })"
                )
                if bbox and isinstance(scroll, dict):
                    box = {
                        "x": float(bbox.get("x", 0)) - float(scroll.get("sx", 0)),
                        "y": float(bbox.get("y", 0)) - float(scroll.get("sy", 0)),
                        "w": float(bbox.get("width", 0)),
                        "h": float(bbox.get("height", 0)),
                        "dpr": float(scroll.get("dpr", 1.0)),
                    }
            except Exception:
                box = None
        try:
            page.wait_for_timeout(350)
        except Exception:
            pass

        png = page.screenshot(full_page=False)
        ctx.close()
        browser.close()

    img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screenshot")

    if box:
        dpr = float(box.get("dpr") or 1.0)
        x = int(max(0, box.get("x", 0) * dpr))
        y = int(max(0, box.get("y", 0) * dpr))
        w = int(max(1, box.get("w", 0) * dpr))
        h = int(max(1, box.get("h", 0) * dpr))
        x2 = min(img.shape[1] - 1, x + w)
        y2 = min(img.shape[0] - 1, y + h)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 3)

    title = str(issue.get("message") or "Issue")
    cv2.rectangle(img, (0, 0), (img.shape[1], 56), (15, 15, 15), -1)
    cv2.putText(img, title[:90], (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return out_path


@app.get("/issue/{job_id}/{idx}", response_class=HTMLResponse)
def issue_detail(request: Request, job_id: str, idx: int) -> HTMLResponse:
    if not _is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if idx < 0 or idx >= len(job.issues):
        raise HTTPException(status_code=404, detail="Issue not found")
    return templates.TemplateResponse(
        "issue.html",
        {
            "request": request,
            "job_id": job_id,
            "idx": idx,
            "issue": job.issues[idx],
        },
    )


@app.get("/shot/{job_id}/{idx}.png")
def issue_screenshot(request: Request, job_id: str, idx: int) -> FileResponse:
    if not _is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if idx < 0 or idx >= len(job.issues):
        raise HTTPException(status_code=404, detail="Issue not found")

    path = _ensure_issue_screenshot(job, idx)
    return FileResponse(str(path), filename=f"{idx}.png", media_type="image/png")


@app.get("/download/{job_id}/{name}")
def download(request: Request, job_id: str, name: str) -> FileResponse:
    if not _is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or not job.out_dir:
        raise HTTPException(status_code=404, detail="Job not found")

    if name not in ("issues.csv", "report.json"):
        raise HTTPException(status_code=404, detail="File not found")

    path = Path(job.out_dir) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(path), filename=name)
