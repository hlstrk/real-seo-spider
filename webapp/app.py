from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os
import runpy
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
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


@dataclass
class User:
    username: str
    password_hash: str
    salt: str
    role: str = "user"


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUT_DIR = Path("out") / "web"
JOBS_DB_PATH = OUT_DIR / "jobs.json"
USERS_DB_PATH = OUT_DIR / "users.json"
SETTINGS_PATH = OUT_DIR / "settings.json"
SHOTS_DIRNAME = "shots"

DEFAULT_RATE_LIMIT = os.getenv("WEBAPP_RATE_LIMIT", "60/minute")
LOGIN_RATE_LIMIT = os.getenv("WEBAPP_LOGIN_RATE_LIMIT", "5/minute")
SCAN_RATE_LIMIT = os.getenv("WEBAPP_SCAN_RATE_LIMIT", "10/hour")
ADMIN_RATE_LIMIT = os.getenv("WEBAPP_ADMIN_RATE_LIMIT", "20/hour")
SESSION_SECRET = os.getenv("WEBAPP_SESSION_SECRET") or secrets.token_hex(32)
DEFAULT_MAX_PAGES_LIMIT = int(os.getenv("WEBAPP_MAX_PAGES_LIMIT", "200"))
DEFAULT_MAX_PAGES_DEFAULT = int(os.getenv("WEBAPP_MAX_PAGES_DEFAULT", "50"))

# Auth (very simple, local use). Configure via env vars for safety.
ADMIN_USERNAME = os.getenv("WEBAPP_ADMIN_USERNAME", "lema")
ADMIN_PASSWORD = (os.getenv("WEBAPP_ADMIN_PASSWORD") or os.getenv("ADMIN_PASSWORD") or "").strip()
ADMIN_PASSWORD_SET = bool(ADMIN_PASSWORD)

app = FastAPI(title="SEO Pulse")
limiter = Limiter(key_func=get_remote_address, default_limits=[DEFAULT_RATE_LIMIT])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["fmt_dt"] = lambda ts: datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")

JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
USERS: Dict[str, User] = {}
USERS_LOCK = threading.Lock()
SETTINGS: Dict[str, int] = {
    "max_pages_limit": max(1, DEFAULT_MAX_PAGES_LIMIT),
    "max_pages_default": max(1, min(DEFAULT_MAX_PAGES_DEFAULT, DEFAULT_MAX_PAGES_LIMIT)),
}

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

SECURITY_HEADERS = [
    "strict-transport-security",
    "content-security-policy",
    "x-content-type-options",
    "referrer-policy",
    "permissions-policy",
    "x-frame-options",
]


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    if not password:
        raise ValueError("Password required")
    salt_hex = salt_hex or secrets.token_hex(16)
    try:
        salt_bytes = bytes.fromhex(salt_hex)
    except ValueError as exc:
        raise ValueError("Invalid salt") from exc
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return salt_hex, digest.hex()


def _verify_password(password: str, user: User) -> bool:
    try:
        salt_bytes = bytes.fromhex(user.salt)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
        return hmac.compare_digest(user.password_hash, digest.hex())
    except Exception:
        return False


def _save_users_db() -> None:
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USERS_LOCK:
        payload = {"users": [dataclasses.asdict(u) for u in USERS.values()]}
    USERS_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_default_admin() -> None:
    if not ADMIN_PASSWORD_SET:
        return
    username = ADMIN_USERNAME or "admin"
    created = False
    with USERS_LOCK:
        if username not in USERS:
            salt, pw_hash = _hash_password(ADMIN_PASSWORD)
            USERS[username] = User(username=username, password_hash=pw_hash, salt=salt, role="admin")
            created = True
    if created:
        _save_users_db()


def _load_users_db() -> None:
    if USERS_DB_PATH.exists():
        try:
            data = json.loads(USERS_DB_PATH.read_text(encoding="utf-8"))
            items = data.get("users", [])
            with USERS_LOCK:
                USERS.clear()
                for it in items:
                    u = User(
                        username=str(it.get("username") or ""),
                        password_hash=str(it.get("password_hash") or ""),
                        salt=str(it.get("salt") or ""),
                        role=str(it.get("role") or "user"),
                    )
                    if u.username and u.password_hash and u.salt:
                        USERS[u.username] = u
        except Exception:
            with USERS_LOCK:
                USERS.clear()
    _ensure_default_admin()


def _get_user(username: str) -> Optional[User]:
    with USERS_LOCK:
        return USERS.get(username)


def _create_user(username: str, password: str, role: str = "user") -> None:
    username = username.strip()
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if role not in ("admin", "user"):
        raise ValueError("Role must be admin or user.")
    with USERS_LOCK:
        if username in USERS:
            raise ValueError("User already exists.")
    salt, pw_hash = _hash_password(password)
    with USERS_LOCK:
        USERS[username] = User(username=username, password_hash=pw_hash, salt=salt, role=role)
    _save_users_db()


def _reset_password(username: str, password: str) -> None:
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with USERS_LOCK:
        user = USERS.get(username)
    if not user:
        raise ValueError("User not found.")
    salt, pw_hash = _hash_password(password)
    with USERS_LOCK:
        user.salt = salt
        user.password_hash = pw_hash
    _save_users_db()


def _load_settings() -> None:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        limit = int(data.get("max_pages_limit") or SETTINGS["max_pages_limit"])
        default = int(data.get("max_pages_default") or SETTINGS["max_pages_default"])
        SETTINGS["max_pages_limit"] = max(1, limit)
        SETTINGS["max_pages_default"] = max(1, min(default, SETTINGS["max_pages_limit"]))
    except Exception:
        SETTINGS["max_pages_limit"] = max(1, SETTINGS["max_pages_limit"])
        SETTINGS["max_pages_default"] = max(1, min(SETTINGS["max_pages_default"], SETTINGS["max_pages_limit"]))


def _save_settings() -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_security_report(url: str, timeout: int = 10) -> Dict[str, Any]:
    import requests
    from bs4 import BeautifulSoup  # type: ignore
    from urllib.parse import urlparse, urljoin

    report: Dict[str, Any] = {
        "url": url,
        "headers": {},
        "missing_headers": [],
        "assets": {"css": [], "js": [], "img": []},
        "insecure_assets": [],
        "asset_origins": {"internal": [], "external": []},
    }
    try:
        parsed = urlparse(url)
        origin_host = (parsed.hostname or "").lower()
        origin_scheme = parsed.scheme or "https"

        def _normalize_link(link: str) -> str:
            if not link:
                return ""
            if link.startswith("//"):
                return f"{origin_scheme}:{link}"
            if link.startswith("http://") or link.startswith("https://"):
                return link
            # relative
            return urljoin(f"{origin_scheme}://{origin_host}", link)

        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        report["headers"] = {k.lower(): v for k, v in resp.headers.items()}

        missing = []
        for h in SECURITY_HEADERS:
            if h not in report["headers"]:
                missing.append(h)
        report["missing_headers"] = missing

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type and resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            css_links = [link.get("href") or "" for link in soup.find_all("link", rel=lambda v: v and "stylesheet" in v)]
            js_links = [script.get("src") or "" for script in soup.find_all("script") if script.get("src")]
            img_links = [img.get("src") or "" for img in soup.find_all("img") if img.get("src")]
            report["assets"]["css"] = css_links
            report["assets"]["js"] = js_links
            report["assets"]["img"] = img_links
            all_assets = css_links + js_links + img_links
            report["insecure_assets"] = [a for a in all_assets if a.startswith("http://")]

            origins: Dict[str, Dict[str, Any]] = {}
            for kind, links in (("css", css_links), ("js", js_links), ("img", img_links)):
                for lk in links:
                    norm = _normalize_link(lk)
                    host = (urlparse(norm).hostname or "").lower()
                    key = host or "(bilinmiyor)"
                    entry = origins.setdefault(
                        key,
                        {"host": key, "count": 0, "types": set(), "sample": norm or lk},
                    )
                    entry["count"] += 1
                    entry["types"].add(kind)
            external = []
            internal = []
            for entry in origins.values():
                entry["types"] = sorted(entry["types"])
                if entry["host"] == origin_host or not entry["host"]:
                    internal.append(entry)
                else:
                    external.append(entry)
            external_sorted = sorted(external, key=lambda x: (-x["count"], x["host"]))
            internal_sorted = sorted(internal, key=lambda x: (-x["count"], x["host"]))
            report["asset_origins"] = {
                "external": external_sorted,
                "internal": internal_sorted,
            }
    except Exception as e:
        report["error"] = str(e)
    return report

def _load_openai_key_from_files() -> str:
    # Priority:
    # 1) env var OPENAI_API_KEY (recommended)
    # 2) config.py (local, not committed)
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


def _current_user(request: Request) -> Optional[User]:
    username = str(request.session.get("username") or "").strip()
    if not username:
        return None
    return _get_user(username)


def _is_logged_in(request: Request) -> bool:
    return _current_user(request) is not None


def _require_login(request: Request) -> None:
    if not _is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")


def _require_admin(request: Request) -> User:
    user = _current_user(request)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    return user


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
        no_gpt_requested = bool(p.get("no_gpt"))
        sleep = float(p.get("sleep") or 0)
        model = str(p.get("model") or (_load_openai_model_from_files() or "gpt-4o-mini"))
        api_key = _load_openai_key_from_files()
        use_gpt = bool(api_key) and not no_gpt_requested

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
        if use_gpt:
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
_load_users_db()
_load_settings()


def _render_admin(
    request: Request, message: str = "", error: str = "", status_code: int = 200
) -> HTMLResponse:
    _require_admin(request)
    with USERS_LOCK:
        users_list = sorted(
            [{"username": u.username, "role": u.role} for u in USERS.values()],
            key=lambda u: (u["role"] != "admin", u["username"]),
        )
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "users": users_list,
            "message": message,
            "error": error,
            "settings": SETTINGS,
        },
        status_code=status_code,
    )


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request) -> HTMLResponse:
    if _is_logged_in(request):
        return RedirectResponse(url="/", status_code=303)
    if not USERS:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Admin hesabı yok. WEBAPP_ADMIN_PASSWORD ile bir admin oluşturup uygulamayı yeniden başlatın.",
            },
            status_code=503,
        )
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/login")
@limiter.limit(LOGIN_RATE_LIMIT)
def login_post(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
):
    if not USERS:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Admin hesabı yok. WEBAPP_ADMIN_PASSWORD ile bir admin oluşturup uygulamayı yeniden başlatın.",
            },
            status_code=503,
        )
    user = _get_user(username)
    if user and _verify_password(password, user):
        request.session.clear()
        request.session["auth"] = "ok"
        request.session["username"] = user.username
        request.session["role"] = user.role
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Hatalı kullanıcı adı veya şifre."})


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request) -> HTMLResponse:
    return _render_admin(request)


@app.post("/admin/users")
@limiter.limit(ADMIN_RATE_LIMIT)
def admin_users(
    request: Request,
    action: str = Form(...),
    username: str = Form(""),
    password: str = Form(""),
    role: str = Form("user"),
) -> HTMLResponse:
    try:
        _require_admin(request)
        if action == "create":
            _create_user(username, password, role)
            return _render_admin(request, message=f"Kullanıcı oluşturuldu: {username.strip()}")
        if action == "reset":
            _reset_password(username, password)
            return _render_admin(request, message=f"Şifre güncellendi: {username.strip()}")
        raise ValueError("Geçersiz işlem.")
    except Exception as e:
        return _render_admin(request, error=str(e), status_code=400)


@app.post("/admin/settings")
@limiter.limit(ADMIN_RATE_LIMIT)
def admin_settings(
    request: Request,
    max_pages_limit: int = Form(...),
    max_pages_default: int = Form(...),
) -> HTMLResponse:
    _require_admin(request)
    try:
        limit_val = max(1, min(int(max_pages_limit), 2000))
        default_val = max(1, min(int(max_pages_default), limit_val))
        SETTINGS["max_pages_limit"] = limit_val
        SETTINGS["max_pages_default"] = default_val
        _save_settings()
        return _render_admin(request, message="Ayarlar güncellendi.")
    except Exception as e:
        return _render_admin(request, error=str(e), status_code=400)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    if not _is_logged_in(request):
        return templates.TemplateResponse(
            "landing.html",
            {
                "request": request,
            },
        )
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
                "max_pages": str(SETTINGS["max_pages_default"]),
                "timeout": "30",
                "use_sitemap": False,
                "follow_redirects": False,
                "no_gpt": False,
                "sleep": "0",
                "model": (_load_openai_model_from_files() or "gpt-4o-mini"),
            },
            "max_pages_limit": SETTINGS["max_pages_limit"],
        },
    )


@app.post("/scan")
@limiter.limit(SCAN_RATE_LIMIT)
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
    try:
        max_pages_val = max(1, min(int(max_pages), SETTINGS["max_pages_limit"]))
    except Exception:
        max_pages_val = SETTINGS["max_pages_default"]

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
            "max_pages": max_pages_val,
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
            "SECURITY_HEADERS": SECURITY_HEADERS,
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


@app.get("/security/{job_id}/report")
def security_report(request: Request, job_id: str):
    if not _is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or not job.pages:
        raise HTTPException(status_code=404, detail="Job not found")
    start_url = str(job.params.get("start_url") or job.pages[0].get("url") or "")
    if not start_url:
        raise HTTPException(status_code=400, detail="No URL to check")
    report = _fetch_security_report(start_url)
    print("[security] report for", start_url, json.dumps(report, ensure_ascii=False)[:500])
    return report
