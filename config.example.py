# Copy this file to `config.py` and paste your keys at the top.
# Or set env vars instead (preferred): OPENAI_API_KEY, OPENAI_MODEL.
# Web UI auth and rate limits are configured via env vars:
#   WEBAPP_ADMIN_USERNAME, WEBAPP_ADMIN_PASSWORD (required for first admin), WEBAPP_SESSION_SECRET
#   WEBAPP_RATE_LIMIT, WEBAPP_LOGIN_RATE_LIMIT, WEBAPP_SCAN_RATE_LIMIT, WEBAPP_ADMIN_RATE_LIMIT
#   WEBAPP_MAX_PAGES_LIMIT, WEBAPP_MAX_PAGES_DEFAULT

OPENAI_API_KEY = ""  # <-- paste your OpenAI API key here if you want
OPENAI_MODEL = "gpt-4o-mini"

BASE_DOMAIN = "lemaclinic.com"
START_URL = "https://lemaclinic.com/"

# Which URL paths are considered "in scope".
# If your English pages live under /en/, set: SCOPE_PREFIXES = ["/en/"]
# If English pages are directly under the domain (no /en), keep the default:
SCOPE_PREFIXES = ["/"]

# Optional: require the page's <html lang="..."> to start with this value (e.g. "en").
# Leave empty to not filter by language attribute.
REQUIRE_LANG = ""

# Crawl limits
MAX_PAGES = 200
REQUEST_TIMEOUT_SECONDS = 30