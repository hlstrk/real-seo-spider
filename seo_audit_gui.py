from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from tkinter import BOTH, DISABLED, END, NORMAL, BooleanVar, StringVar, Text, Tk, filedialog, messagebox
from tkinter import ttk


@dataclass(frozen=True)
class RunConfig:
    start_url: str
    domain: str
    mode: str
    fetcher: str
    max_pages: str
    timeout: str
    out_csv: str
    out_json: str
    scope_prefixes: str
    require_lang: str
    no_gpt: bool
    follow_redirects: bool
    use_sitemap: bool
    sleep: str
    model: str
    api_key: str
    headed: bool


class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("real-seo-spider (lemaclinic)")
        self.root.minsize(900, 680)

        self._proc: subprocess.Popen[str] | None = None
        self._q: "queue.Queue[str]" = queue.Queue()

        self.var_start_url = StringVar(value="https://lemaclinic.com/")
        self.var_domain = StringVar(value="lemaclinic.com")
        self.var_mode = StringVar(value="single")
        self.var_fetcher = StringVar(value="requests")
        self.var_max_pages = StringVar(value="200")
        self.var_timeout = StringVar(value="30")
        self.var_out_csv = StringVar(value=os.path.join("out", "issues.csv"))
        self.var_out_json = StringVar(value=os.path.join("out", "report.json"))
        self.var_scope_prefixes = StringVar(value="/")
        self.var_require_lang = StringVar(value="")
        self.var_sleep = StringVar(value="0")
        self.var_model = StringVar(value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.var_api_key = StringVar(value=os.getenv("OPENAI_API_KEY", ""))

        self.var_no_gpt = BooleanVar(value=False)
        self.var_follow_redirects = BooleanVar(value=False)
        self.var_use_sitemap = BooleanVar(value=False)
        self.var_headed = BooleanVar(value=False)

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=BOTH, expand=True)

        top = ttk.LabelFrame(outer, text="Ayarlar", padding=10)
        top.pack(fill="x")

        def add_row(row: int, label: str, widget: ttk.Widget) -> None:
            ttk.Label(top, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

        top.columnconfigure(1, weight=1)

        add_row(0, "Start URL", ttk.Entry(top, textvariable=self.var_start_url))
        add_row(1, "Domain", ttk.Entry(top, textvariable=self.var_domain))

        mode_cb = ttk.Combobox(top, textvariable=self.var_mode, values=["single", "crawl"], state="readonly")
        fetcher_cb = ttk.Combobox(top, textvariable=self.var_fetcher, values=["requests", "chromium"], state="readonly")
        add_row(2, "Mode", mode_cb)
        add_row(3, "Fetcher", fetcher_cb)

        add_row(4, "Max Pages (crawl)", ttk.Entry(top, textvariable=self.var_max_pages))
        add_row(5, "Timeout (sec)", ttk.Entry(top, textvariable=self.var_timeout))
        add_row(6, "Scope Prefix(es) (comma)", ttk.Entry(top, textvariable=self.var_scope_prefixes))
        add_row(7, "Require <html lang>", ttk.Entry(top, textvariable=self.var_require_lang))

        out_row = ttk.Frame(top)
        ttk.Entry(out_row, textvariable=self.var_out_csv).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="CSV seç", command=self._pick_csv).pack(side="left", padx=(8, 0))
        add_row(8, "Output CSV", out_row)

        out_row2 = ttk.Frame(top)
        ttk.Entry(out_row2, textvariable=self.var_out_json).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row2, text="JSON seç", command=self._pick_json).pack(side="left", padx=(8, 0))
        add_row(9, "Output JSON", out_row2)

        gpt_row = ttk.Frame(top)
        ttk.Checkbutton(gpt_row, text="GPT kapalı (--no-gpt)", variable=self.var_no_gpt).pack(side="left")
        ttk.Label(gpt_row, text="Sleep (sec)").pack(side="left", padx=(20, 6))
        ttk.Entry(gpt_row, width=6, textvariable=self.var_sleep).pack(side="left")
        add_row(10, "GPT", gpt_row)

        add_row(11, "Model", ttk.Entry(top, textvariable=self.var_model))

        api_row = ttk.Frame(top)
        ttk.Entry(api_row, textvariable=self.var_api_key, show="•").pack(side="left", fill="x", expand=True)
        ttk.Button(api_row, text="Env’den çek", command=self._load_env_keys).pack(side="left", padx=(8, 0))
        add_row(12, "OPENAI_API_KEY", api_row)

        flags = ttk.Frame(top)
        ttk.Checkbutton(flags, text="Follow redirects (single)", variable=self.var_follow_redirects).pack(side="left")
        ttk.Checkbutton(flags, text="Use sitemap (crawl)", variable=self.var_use_sitemap).pack(side="left", padx=(16, 0))
        ttk.Checkbutton(flags, text="Headed Chromium", variable=self.var_headed).pack(side="left", padx=(16, 0))
        add_row(13, "Opsiyonlar", flags)

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(10, 8))

        self.btn_run = ttk.Button(btns, text="Çalıştır", command=self._on_run)
        self.btn_stop = ttk.Button(btns, text="Durdur", command=self._on_stop, state=DISABLED)
        self.btn_open = ttk.Button(btns, text="out/ klasörünü aç", command=self._open_out_dir)
        self.btn_run.pack(side="left")
        self.btn_stop.pack(side="left", padx=(8, 0))
        self.btn_open.pack(side="left", padx=(8, 0))

        log_frame = ttk.LabelFrame(outer, text="Log", padding=8)
        log_frame.pack(fill=BOTH, expand=True)

        self.txt = Text(log_frame, height=18)
        self.txt.pack(fill=BOTH, expand=True)

    def _pick_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=os.path.basename(self.var_out_csv.get() or "issues.csv"),
        )
        if path:
            self.var_out_csv.set(path)

    def _pick_json(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=os.path.basename(self.var_out_json.get() or "report.json"),
        )
        if path:
            self.var_out_json.set(path)

    def _load_env_keys(self) -> None:
        self.var_api_key.set(os.getenv("OPENAI_API_KEY", ""))
        self.var_model.set(os.getenv("OPENAI_MODEL", self.var_model.get()))

    def _append_log(self, line: str) -> None:
        self.txt.insert(END, line)
        if not line.endswith("\n"):
            self.txt.insert(END, "\n")
        self.txt.see(END)

    def _read_proc(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._q.put(line)
        rc = proc.wait()
        self._q.put(f"\n[done] exit code: {rc}\n")
        self._q.put("__PROC_DONE__")

    def _poll_queue(self) -> None:
        try:
            while True:
                line = self._q.get_nowait()
                if line == "__PROC_DONE__":
                    self._proc = None
                    self.btn_run.config(state=NORMAL)
                    self.btn_stop.config(state=DISABLED)
                else:
                    self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _collect_config(self) -> RunConfig:
        return RunConfig(
            start_url=self.var_start_url.get().strip(),
            domain=self.var_domain.get().strip(),
            mode=self.var_mode.get().strip(),
            fetcher=self.var_fetcher.get().strip(),
            max_pages=self.var_max_pages.get().strip(),
            timeout=self.var_timeout.get().strip(),
            out_csv=self.var_out_csv.get().strip(),
            out_json=self.var_out_json.get().strip(),
            scope_prefixes=self.var_scope_prefixes.get().strip(),
            require_lang=self.var_require_lang.get().strip(),
            no_gpt=bool(self.var_no_gpt.get()),
            follow_redirects=bool(self.var_follow_redirects.get()),
            use_sitemap=bool(self.var_use_sitemap.get()),
            sleep=self.var_sleep.get().strip(),
            model=self.var_model.get().strip(),
            api_key=self.var_api_key.get().strip(),
            headed=bool(self.var_headed.get()),
        )

    def _build_cmd(self, cfg: RunConfig) -> list[str]:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "seo_audit.py"),
            "--start",
            cfg.start_url,
            "--domain",
            cfg.domain,
            "--mode",
            cfg.mode,
            "--fetcher",
            cfg.fetcher,
            "--timeout",
            cfg.timeout,
            "--out-csv",
            cfg.out_csv,
            "--out-json",
            cfg.out_json,
            "--sleep",
            cfg.sleep,
        ]

        if cfg.require_lang:
            cmd += ["--require-lang", cfg.require_lang]

        prefixes = [p.strip() for p in (cfg.scope_prefixes or "").split(",") if p.strip()]
        for pfx in prefixes:
            cmd += ["--scope-prefix", pfx]

        if cfg.mode == "crawl":
            cmd += ["--max-pages", cfg.max_pages]
            if cfg.use_sitemap:
                cmd += ["--use-sitemap"]
        else:
            if cfg.follow_redirects:
                cmd += ["--follow-redirects"]

        if cfg.no_gpt:
            cmd += ["--no-gpt"]
        if cfg.headed:
            cmd += ["--headed"]

        return cmd

    def _on_run(self) -> None:
        if self._proc is not None:
            messagebox.showinfo("Çalışıyor", "Bir işlem zaten çalışıyor.")
            return

        cfg = self._collect_config()
        if not cfg.start_url:
            messagebox.showerror("Hata", "Start URL boş olamaz.")
            return

        cmd = self._build_cmd(cfg)
        env = os.environ.copy()
        if cfg.api_key:
            env["OPENAI_API_KEY"] = cfg.api_key
        if cfg.model:
            env["OPENAI_MODEL"] = cfg.model

        self._append_log("Running:\n  " + " ".join(cmd) + "\n")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except Exception as e:
            messagebox.showerror("Başlatılamadı", str(e))
            return

        self._proc = proc
        self.btn_run.config(state=DISABLED)
        self.btn_stop.config(state=NORMAL)
        threading.Thread(target=self._read_proc, args=(proc,), daemon=True).start()

    def _on_stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass

    def _open_out_dir(self) -> None:
        out_dir = os.path.abspath("out")
        os.makedirs(out_dir, exist_ok=True)
        try:
            os.startfile(out_dir)  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("Açılamadı", str(e))


def main() -> int:
    root = Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

