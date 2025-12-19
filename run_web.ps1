$ErrorActionPreference = "Stop"

# Local LAN publish: listen on all interfaces so you can open it from http://192.168.1.100:8000
$env:PYTHONUTF8 = "1"
# Optional: set a stable session secret
# $env:WEBAPP_SESSION_SECRET = "change-me"

python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000
# If you see noisy access logs, you can use:
# python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000 --no-access-log
