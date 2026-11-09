#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cat's diffusion bee 0.1.3 — A.C Holdings / Team Flames
DiffusionBee-style single-file FLUX studio (tkinter GUI + headless CLI).

Engines (local repos are UNGATED mirrors — no HF login / token / license click)
  FLUX.1  local  : unsloth/FLUX.1-schnell | unsloth/FLUX.1-dev     (diffusers FluxPipeline / FluxImg2ImgPipeline)
  FLUX.2  local  : black-forest-labs/FLUX.2-klein-4B (Apache-2.0)  (diffusers Flux2KleinPipeline, 15 GB, Mac-friendly)
                   unsloth/FLUX.2-dev                              (diffusers Flux2Pipeline, 32B — big iron only)
  FLUX.2  cloud  : flux-2-pro / flux-2-max / flux-2-flex / flux-2-klein via api.bfl.ai
  FLUX.3  cloud  : api.bfl.ai, slug editable (early-access; IDs not stable yet)

pip install --upgrade torch "diffusers>=0.37" transformers accelerate sentencepiece protobuf pillow certifi "huggingface_hub[hf_xet]"
Download sizes (one-time, cached in ~/.cache/huggingface): schnell/dev ≈ 31 GB · klein 4B ≈ 15 GB

GUI : python diffusionbee.py
CLI : python diffusionbee.py -p "a cat at a CRT" -m schnell -o cat.png
      python diffusionbee.py -p "a cat at a CRT" -m flux2-pro --key bfl_xxx
"""
import argparse, base64, io, json, os, queue, random, ssl, sys, threading, time, traceback
import urllib.error, urllib.request

try:
    from PIL import Image
except ImportError:
    print("missing pillow → pip install pillow"); sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk
except Exception:  # headless / no Tk build — CLI still works
    tk = ttk = filedialog = messagebox = ImageTk = None

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")   # max parallel chunk downloads
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # our own byte meter replaces tqdm spam
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)

APP = "cat's diffusion bee"
VER = "0.1.3"
HOME = os.path.join(os.path.expanduser("~"), "CatsDiffusionBee")
OUT = os.path.join(HOME, "outputs")
CFG = os.path.join(HOME, "config.json")
HIST = os.path.join(HOME, "history.json")
os.makedirs(OUT, exist_ok=True)

# ── theme: black buttons, blue-hue text ──────────────────────────────
BG      = "#05070d"
PANEL   = "#0b0f1a"
FIELD   = "#000000"
BTN     = "#000000"
BTN_HOV = "#0a1428"
BLUE    = "#4da3ff"
BLUE_HI = "#8cc8ff"
BLUE_LO = "#2a5c99"
BORDER  = "#16325c"
FONT    = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)
FONT_H  = (FONT[0], 15, "bold")
MONO    = ("Consolas", 9) if sys.platform == "win32" else ("Courier", 9)

# ── model registry ───────────────────────────────────────────────────
MODELS = {
    # repos = ungated mirrors first; tried in order until one loads
    "FLUX.1 [schnell] · local": dict(kind="local1", cls="FluxPipeline",
                                    repos=["unsloth/FLUX.1-schnell", "YuCollection/FLUX.1-schnell-Diffusers",
                                           "lzyvegetable/FLUX.1-schnell"],
                                    steps=4, guidance=0.0, max_seq=256, ram=34),
    "FLUX.1 [dev] · local":     dict(kind="local1", cls="FluxPipeline",
                                    repos=["unsloth/FLUX.1-dev"],
                                    steps=28, guidance=3.5, max_seq=512, ram=34),
    "FLUX.2 [klein 4B] · local": dict(kind="local2", cls="Flux2KleinPipeline",
                                    repos=["black-forest-labs/FLUX.2-klein-4B", "unsloth/FLUX.2-klein-4B"],
                                    steps=4, guidance=1.0, ram=18),
    "FLUX.2 [dev] · local":     dict(kind="local2", cls="Flux2Pipeline",
                                    repos=["unsloth/FLUX.2-dev"],
                                    steps=28, guidance=4.0, ram=80),
    "FLUX.2 [pro] · cloud":     dict(kind="api", slug="flux-2-pro",   steps=None, guidance=None),
    "FLUX.2 [max] · cloud":     dict(kind="api", slug="flux-2-max",   steps=None, guidance=None),
    "FLUX.2 [flex] · cloud":    dict(kind="api", slug="flux-2-flex",  steps=28,   guidance=4.5),
    "FLUX.2 [klein] · cloud":   dict(kind="api", slug="flux-2-klein", steps=None, guidance=None),
    "FLUX.3 · cloud":           dict(kind="api", slug="flux-3",       steps=None, guidance=None,
                                    editable_slug=True),
}
ALIASES = {"schnell": "FLUX.1 [schnell] · local", "dev": "FLUX.1 [dev] · local",
           "klein": "FLUX.2 [klein 4B] · local", "flux2": "FLUX.2 [dev] · local", "flux2-dev": "FLUX.2 [dev] · local",
           "flux2-pro": "FLUX.2 [pro] · cloud", "flux2-max": "FLUX.2 [max] · cloud",
           "flux2-flex": "FLUX.2 [flex] · cloud", "flux2-klein": "FLUX.2 [klein] · cloud",
           "flux3": "FLUX.3 · cloud"}
SIZES = ["512x512", "768x768", "1024x1024", "1024x768", "768x1024",
         "1152x896", "896x1152", "1344x768", "768x1344", "1536x640"]
DEFAULT_CFG = {"bfl_key": "", "hf_token": "", "bfl_base": "https://api.bfl.ai",
               "offload": True, "flux3_slug": "flux-3"}


def load_json(p, d):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return d


def save_json(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        try:
            import ctypes
            class M(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("p", ctypes.c_ulong), ("t", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
            m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.t / 2**30
        except Exception:
            return 0.0


def default_model():
    # schnell needs ~34 GB resident; below that it swaps to death → klein 4B
    r = ram_gb()
    return "FLUX.2 [klein 4B] · local" if 0 < r < 40 and sys.platform == "darwin" else "FLUX.1 [schnell] · local"


def fmt_gb(b):
    return f"{b / 2**30:.1f} GB"


def fetch_repo(repo, token, stat, cancel):
    """Download only the diffusers component folders, with live byte progress. Returns local path."""
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download, constants
    idx = json.load(open(hf_hub_download(repo, "model_index.json", token=token)))
    comps = [k for k, v in idx.items() if isinstance(v, list) and v and v[0] is not None]
    pats = ["model_index.json"] + [f"{c}/*" for c in comps]
    files = HfApi().list_repo_tree(repo, recursive=True, token=token)
    total = sum(getattr(f, "size", 0) or 0 for f in files
                if getattr(f, "size", None) is not None and
                (f.path == "model_index.json" or f.path.split("/")[0] in comps))
    blobs = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "blobs")
    out, err = {}, {}

    def work():
        try:
            out["path"] = snapshot_download(repo, allow_patterns=pats, token=token)
        except Exception as e:
            err["e"] = e
    th = threading.Thread(target=work, daemon=True)
    th.start()
    t0, last_b, last_t, speed = time.time(), None, time.time(), 0.0
    while th.is_alive():
        if cancel.is_set():
            raise RuntimeError("cancelled (download resumes next time)")
        th.join(0.5)
        try:
            have = sum(e.stat().st_size for e in os.scandir(blobs) if e.is_file())
        except FileNotFoundError:
            have = 0
        now = time.time()
        if last_b is not None and now > last_t:
            inst = (have - last_b) / (now - last_t)
            speed = inst if speed == 0 else 0.85 * speed + 0.15 * inst
        last_b, last_t = have, now
        have = min(have, total) if total else have
        eta = (total - have) / speed if speed > 1 else 0
        stat(f"downloading {repo}  {fmt_gb(have)} / {fmt_gb(total)}  {speed / 2**20:5.1f} MB/s"
             + (f"  ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta else ""),
             have / total if total else 0.0)
    if "e" in err:
        raise err["e"]
    stat(f"downloaded {repo} ({fmt_gb(total)}) in {time.time() - t0:.0f}s", 1.0)
    return out["path"]


def ssl_ctx():
    # macOS python.org builds ship without a CA store → CERTIFICATE_VERIFY_FAILED
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def friendly(e):
    """Turn the usual tracebacks into one actionable line."""
    s, n = str(e), type(e).__name__
    if isinstance(e, ModuleNotFoundError):
        return (f"missing module '{e.name}' → pip install -U torch diffusers transformers "
                f"accelerate sentencepiece protobuf")
    if "GatedRepo" in n or "gated repo" in s.lower() or "401 Client Error" in s:
        return "all mirrors unreachable → check internet / disk space, or pick FLUX.1 [schnell] / klein"
    if "out of memory" in s.lower() or "OutOfMemory" in n or "DefaultCPUAllocator" in s:
        return "out of memory → enable CPU offload, use 512x512, or switch to FLUX.1 [schnell] / a cloud model"
    if "CERTIFICATE_VERIFY_FAILED" in s:
        return "SSL cert error → pip install certifi (macOS: run 'Install Certificates.command')"
    return f"{n}: {s}"


# ── engines ──────────────────────────────────────────────────────────
class LocalEngine:
    """Lazy diffusers loader; caches one pipeline at a time to keep VRAM sane."""

    def __init__(self, log):
        self.log = log
        self.key = None
        self.pipe = None
        self.torch = None
        self.stat = lambda text, frac: None
        self.cancel = threading.Event()

    def device(self):
        t = self.torch
        if t.cuda.is_available():
            return "cuda"
        if getattr(t.backends, "mps", None) and t.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, spec, img2img, offload, hf_token):
        import torch
        self.torch = torch
        key = (spec["cls"], img2img, offload)
        if key == self.key and self.pipe is not None:
            return self.pipe
        self.unload()
        import diffusers
        try:
            diffusers.utils.logging.disable_progress_bar()
        except Exception:
            pass
        name = "FluxImg2ImgPipeline" if (spec["kind"] == "local1" and img2img) else spec["cls"]
        cls = getattr(diffusers, name, None)
        if cls is None:
            raise RuntimeError(f"diffusers {diffusers.__version__} has no {name} → pip install -U \"diffusers>=0.37\"")
        dev = self.device()
        # FLUX in fp16 overflows → NaN → black images. bf16 everywhere (MPS needs macOS 14+).
        dtype = torch.bfloat16
        tok = hf_token or False   # False = ignore any cached HF login
        need = spec.get("ram", 0)
        if dev in ("mps", "cpu") and need and 0 < ram_gb() < need:
            self.log(f"WARNING: {ram_gb():.0f} GB RAM < ~{need} GB this model needs → it will swap hard. "
                     f"Use FLUX.2 [klein 4B] (-m klein).")
        pipe, last = None, None
        for repo in spec["repos"]:
            try:
                path = fetch_repo(repo, tok, self.stat, self.cancel)
                self.log(f"loading {repo} into memory ({name}, {dev}, bf16)…")
                self.stat(f"loading weights ({dev})…", 1.0)
                pipe = cls.from_pretrained(path, torch_dtype=dtype)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if "cancelled" in msg:
                    raise
                if any(t in msg for t in ("Gated", "gated", "401", "403", "404", "RepositoryNotFound",
                                          "Cannot access", "does not appear to have", "EntryNotFound")):
                    self.log(f"  {repo} unavailable → next mirror")
                    last = e
                    continue
                raise
        if pipe is None:
            raise last or RuntimeError("no loadable repo")
        if dev == "cuda" and offload:
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            if vram < 30:
                self.log(f"{vram:.0f} GB VRAM → sequential CPU offload (slow, but fits)")
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to(dev)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        if dev == "mps":
            try:
                pipe.enable_attention_slicing()   # big peak-memory win on unified memory
            except Exception:
                pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self.pipe, self.key = pipe, key
        self.log("model ready.")
        return pipe

    def unload(self):
        if self.pipe is not None:
            self.pipe, self.key = None, None
            try:
                import gc; gc.collect()
                if self.torch and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception:
                pass

    def run(self, spec, p, progress, cancel, stat=None):
        img2img = p["init_image"] is not None
        self.cancel = cancel
        if stat:
            self.stat = stat
        pipe = self.load(spec, img2img and spec["kind"] == "local1", p["offload"], p["hf_token"])
        torch = self.torch
        steps = p["steps"]
        # img2img only runs int(steps*strength) denoise steps
        total = max(1, int(steps * p["strength"])) if (img2img and spec["kind"] == "local1") else steps

        def cb(_pipe, step, _t, kwargs):
            if cancel.is_set():
                _pipe._interrupt = True
            progress(step + 1, total)
            return kwargs

        out = []
        for i in range(p["count"]):
            if cancel.is_set():
                break
            seed = (p["seed"] + i) % 2**32
            gen = torch.Generator("cpu").manual_seed(seed)
            kw = dict(prompt=p["prompt"], num_inference_steps=steps, guidance_scale=p["guidance"],
                      generator=gen, callback_on_step_end=cb, height=p["h"], width=p["w"])
            if spec["kind"] == "local1":
                kw["max_sequence_length"] = spec.get("max_seq", 512)
                if img2img:
                    kw["image"] = p["init_image"].convert("RGB").resize((p["w"], p["h"]))
                    kw["strength"] = p["strength"]
            elif img2img:
                kw["image"] = [p["init_image"].convert("RGB")]
            img = pipe(**kw).images[0]
            out.append((img, seed))
        return out


class BFLEngine:
    """api.bfl.ai: POST /v1/{slug} → poll polling_url → download result.sample."""

    def __init__(self, log):
        self.log = log
        self.ctx = ssl_ctx()

    def _req(self, url, key, data=None):
        h = {"accept": "application/json", "x-key": key}
        body = None
        if data is not None:
            h["Content-Type"] = "application/json"
            body = json.dumps(data).encode()
        r = urllib.request.Request(url, data=body, headers=h, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(r, timeout=60, context=self.ctx) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            hint = {401: "bad API key", 402: "out of credits", 404: "unknown endpoint slug",
                    422: "bad params", 429: "rate limited"}.get(e.code, "")
            raise RuntimeError(f"BFL HTTP {e.code} {hint}: {detail}") from None

    def run(self, spec, p, progress, cancel, stat=None):
        key = p["bfl_key"] or os.environ.get("BFL_API_KEY", "")
        if not key:
            raise RuntimeError("no BFL API key (Settings tab, --key, or $BFL_API_KEY)")
        slug = p["slug"] or spec["slug"]
        base = p["bfl_base"].rstrip("/")
        out = []
        for i in range(p["count"]):
            if cancel.is_set():
                break
            seed = (p["seed"] + i) % 2**32
            body = dict(prompt=p["prompt"], width=p["w"], height=p["h"], seed=seed, output_format="png")
            if spec.get("steps"):
                body["steps"] = p["steps"]
                body["guidance"] = p["guidance"]
            if p["init_image"] is not None:
                buf = io.BytesIO(); p["init_image"].convert("RGB").save(buf, "PNG")
                body["input_image"] = base64.b64encode(buf.getvalue()).decode()
            self.log(f"POST {base}/v1/{slug}")
            job = self._req(f"{base}/v1/{slug}", key, body)
            poll = job.get("polling_url") or f"{base}/v1/get_result?id={job['id']}"
            t0, tick = time.time(), 0
            while True:
                if cancel.is_set():
                    return out
                time.sleep(0.6)
                tick += 1
                progress(min(tick, 95), 100)
                st = self._req(poll, key)
                s = st.get("status")
                if s == "Ready":
                    url = st["result"]["sample"]
                    with urllib.request.urlopen(url, timeout=120, context=self.ctx) as r:
                        img = Image.open(io.BytesIO(r.read())).convert("RGB")
                    out.append((img, seed))
                    progress(100, 100)
                    break
                if s in ("Error", "Failed", "Request Moderated", "Content Moderated", "Task not found"):
                    raise RuntimeError(f"BFL: {s} {st.get('details', '')}")
                if time.time() - t0 > 600:
                    raise RuntimeError("BFL: timeout")
        return out


def save_output(img, seed, out_path=None):
    fn = out_path or os.path.join(OUT, f"cdb_{int(time.time() * 1000)}_{seed}.png")
    d = os.path.dirname(os.path.abspath(fn))
    os.makedirs(d, exist_ok=True)
    img.save(fn)
    return fn


# ── headless CLI ─────────────────────────────────────────────────────
def cli(argv):
    ap = argparse.ArgumentParser(prog="diffusionbee.py", description=f"{APP} {VER} — headless")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-m", "--model", default="auto", help="auto|" + "|".join(ALIASES))
    ap.add_argument("-o", "--out", default=None, help="output .png (default ~/CatsDiffusionBee/outputs)")
    ap.add_argument("-W", "--width", type=int, default=1024)
    ap.add_argument("-H", "--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("--init", default=None, help="input image for img2img")
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--key", default=None, help="BFL API key")
    ap.add_argument("--no-offload", action="store_true")
    a = ap.parse_args(argv)
    name = default_model() if a.model == "auto" else ALIASES.get(a.model, a.model)
    if name not in MODELS:
        ap.error(f"unknown model {a.model!r}; pick: {', '.join(ALIASES)}")
    spec = MODELS[name]
    cfg = {**DEFAULT_CFG, **load_json(CFG, {})}
    seed = a.seed if a.seed >= 0 else random.randint(0, 2**32 - 1)
    p = dict(prompt=a.prompt, w=a.width // 16 * 16, h=a.height // 16 * 16,
             steps=a.steps or spec.get("steps") or 28,
             guidance=a.guidance if a.guidance is not None else (spec.get("guidance") or 0.0),
             seed=seed, count=max(1, a.count), strength=a.strength,
             init_image=Image.open(a.init) if a.init else None,
             offload=not a.no_offload and cfg.get("offload", True), hf_token=cfg.get("hf_token", ""),
             bfl_key=a.key or cfg.get("bfl_key", ""), bfl_base=cfg.get("bfl_base", "https://api.bfl.ai"),
             slug=cfg.get("flux3_slug", "flux-3") if spec.get("editable_slug") else None)
    log = lambda s: print(f"[bee] {s}", flush=True)

    def progress(i, n):
        pct = int(i / max(n, 1) * 100)
        print(f"\r[bee] {'#' * (pct // 4):<25} {pct:3d}%", end="", flush=True)

    def stat(text, frac):
        print(f"\r[bee] {text:<100}", end="", flush=True)
        if frac >= 1.0:
            print()

    eng = BFLEngine(log) if spec["kind"] == "api" else LocalEngine(log)
    log(f"{name} · {p['w']}x{p['h']} · seed {seed}")
    try:
        res = eng.run(spec, p, progress, threading.Event(), stat)
    except Exception as e:
        print()
        log("ERROR " + friendly(e))
        if os.environ.get("BEE_DEBUG"):
            traceback.print_exc()
        return 1
    print()
    for i, (img, s) in enumerate(res):
        out = a.out
        if out and len(res) > 1:
            root, ext = os.path.splitext(out); out = f"{root}_{i}{ext or '.png'}"
        log(f"saved → {save_output(img, s, out)}")
    return 0


# ── GUI ──────────────────────────────────────────────────────────────
_Label = tk.Label if tk else object
_Tk = tk.Tk if tk else object


class BeeButton(_Label):
    def __init__(self, master, text, cmd, big=False, **kw):
        super().__init__(master, text=text, bg=BTN, fg=BLUE, cursor="hand2",
                         font=(FONT[0], 12 if big else 10, "bold"),
                         padx=18 if big else 12, pady=10 if big else 5,
                         highlightthickness=1, highlightbackground=BORDER, **kw)
        self.cmd, self.enabled = cmd, True
        self.bind("<Enter>", lambda e: self.enabled and self.config(bg=BTN_HOV, fg=BLUE_HI))
        self.bind("<Leave>", lambda e: self.config(bg=BTN, fg=BLUE if self.enabled else BLUE_LO))
        self.bind("<Button-1>", lambda e: self.enabled and self.cmd())

    def set_enabled(self, on):
        self.enabled = on
        self.config(fg=BLUE if on else BLUE_LO, cursor="hand2" if on else "arrow")


def lbl(m, t, **kw):
    return tk.Label(m, text=t, bg=kw.pop("bg", PANEL), fg=kw.pop("fg", BLUE), font=kw.pop("font", FONT), **kw)


def entry(m, var, w=10, show=None):
    return tk.Entry(m, textvariable=var, width=w, bg=FIELD, fg=BLUE_HI, insertbackground=BLUE,
                    relief="flat", highlightthickness=1, highlightbackground=BORDER,
                    highlightcolor=BLUE, font=FONT, show=show)


def spin(m, var, value, **kw):
    # tk.Spinbox stomps its textvariable with from_ on creation → set the value after
    s = tk.Spinbox(m, textvariable=var, bg=FIELD, fg=BLUE_HI, buttonbackground=BTN, relief="flat",
                   insertbackground=BLUE, disabledbackground=FIELD, disabledforeground=BLUE_LO, **kw)
    var.set(value)
    return s


def num(var, cast, default):
    try:
        return cast(var.get())
    except Exception:
        return default


class CatsDiffusionBee(_Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP} {VER}")
        self.geometry("1280x820")
        self.minsize(1000, 680)
        self.configure(bg=BG)
        self.cfg = {**DEFAULT_CFG, **load_json(CFG, {})}
        self.history = load_json(HIST, [])
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.current = "Text to Image"
        self.thumbs = []
        self.local = LocalEngine(self.log)
        self.api = BFLEngine(self.log)
        self._style()
        self._build()
        self.after(50, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.report_callback_exception = self._tk_error

    def _tk_error(self, *exc):
        self._log_now("".join(traceback.format_exception(*exc)))

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=FIELD, background=BTN, foreground=BLUE_HI,
                    arrowcolor=BLUE, bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", BLUE_HI)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", BLUE_HI)])
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", BLUE_HI)
        self.option_add("*TCombobox*Listbox.selectBackground", BTN_HOV)
        self.option_add("*TCombobox*Listbox.selectForeground", BLUE_HI)
        s.configure("Bee.Horizontal.TProgressbar", troughcolor=FIELD, background=BLUE,
                    bordercolor=BORDER, lightcolor=BLUE, darkcolor=BLUE_LO)

    def _build(self):
        side = tk.Frame(self, bg=PANEL, width=190, highlightthickness=1, highlightbackground=BORDER)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        # no emoji: color-emoji glyphs crash Tk on X11 (BadLength) and Tk < 8.6.10
        lbl(side, "cat's", font=FONT_H).pack(pady=(18, 0))
        lbl(side, "diffusion bee", font=FONT_H).pack()
        lbl(side, f"v{VER} · FLUX 1 · 2 · 3", fg=BLUE_LO).pack(pady=(0, 18))
        self.pages, self.navbtn = {}, {}
        body = tk.Frame(self, bg=BG)
        body.pack(side="left", fill="both", expand=True)
        for name in ("Text to Image", "Image to Image", "History", "Settings"):
            b = BeeButton(side, name, lambda n=name: self.show(n))
            b.pack(fill="x", padx=12, pady=4)
            self.navbtn[name] = b
            self.pages[name] = tk.Frame(body, bg=BG)
        self.status = lbl(side, "idle", fg=BLUE_LO, wraplength=170, justify="left")
        self.status.pack(side="bottom", pady=12, padx=10)

        self._gen_page(self.pages["Text to Image"], img2img=False)
        self._gen_page(self.pages["Image to Image"], img2img=True)
        self._history_page(self.pages["History"])
        self._settings_page(self.pages["Settings"])
        self.show("Text to Image")

    def show(self, name):
        for n, f in self.pages.items():
            f.pack_forget()
            self.navbtn[n].config(highlightbackground=BLUE if n == name else BORDER)
        self.pages[name].pack(fill="both", expand=True)
        self.current = name
        if name == "History":
            self._refresh_history()

    def _gen_page(self, page, img2img):
        dm = default_model()
        first = MODELS[dm]
        v = dict(model=tk.StringVar(value=dm), size=tk.StringVar(value="1024x1024"),
                 steps=tk.IntVar(), guidance=tk.DoubleVar(), seed=tk.StringVar(value="-1"),
                 count=tk.IntVar(), strength=tk.DoubleVar(value=0.75))
        top = tk.Frame(page, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        prompt = tk.Text(top, height=4, bg=FIELD, fg=BLUE_HI, insertbackground=BLUE, wrap="word",
                         relief="flat", highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=BLUE, font=(FONT[0], 11))
        prompt.pack(side="left", fill="x", expand=True)
        prompt.insert("1.0", "a cozy pixel-art cat hacker at a CRT, neon blue glow, 16-bit")
        prompt.bind("<Control-Return>", lambda e: (self.generate(page), "break")[1])
        gen = BeeButton(top, "Generate", lambda: self.generate(page), big=True)
        gen.pack(side="left", padx=(10, 0), fill="y")

        opts = tk.Frame(page, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        opts.pack(fill="x", padx=16, pady=4)
        r = [0]

        def row(label, w, col):
            lbl(opts, label).grid(row=r[0], column=col, sticky="e", padx=(10, 4), pady=6)
            w.grid(row=r[0], column=col + 1, sticky="w", padx=(0, 10), pady=6)

        cb = ttk.Combobox(opts, textvariable=v["model"], values=list(MODELS), state="readonly", width=26)
        row("Model", cb, 0)
        row("Size", ttk.Combobox(opts, textvariable=v["size"], values=SIZES, width=11), 2)
        row("Images", spin(opts, v["count"], 1, from_=1, to=8, width=4), 4)
        r[0] = 1
        stp = spin(opts, v["steps"], first["steps"], from_=1, to=100, width=5)
        row("Steps", stp, 0)
        gui = spin(opts, v["guidance"], first["guidance"], from_=0.0, to=30.0, increment=0.5, width=5)
        row("Guidance", gui, 2)
        sf = tk.Frame(opts, bg=PANEL)
        entry(sf, v["seed"], 12).pack(side="left")
        BeeButton(sf, "rnd", lambda: v["seed"].set(str(random.randint(0, 2**32 - 1)))).pack(side="left", padx=4)
        row("Seed", sf, 4)
        page.init_lbl = None
        if img2img:
            r[0] = 2
            sc = tk.Scale(opts, from_=0.05, to=1.0, resolution=0.05, orient="horizontal", variable=v["strength"],
                          bg=PANEL, fg=BLUE, troughcolor=FIELD, highlightthickness=0, length=180,
                          activebackground=BLUE)
            v["strength"].set(0.75)
            row("Strength", sc, 0)
            pick = tk.Frame(opts, bg=PANEL)
            BeeButton(pick, "Load input image...", lambda: self._pick_init(page)).pack(side="left")
            page.init_lbl = lbl(pick, "none", fg=BLUE_LO)
            page.init_lbl.pack(side="left", padx=8)
            row("Input", pick, 2)

        def on_model(*_):
            s = MODELS[v["model"].get()]
            if s.get("steps") is not None:
                stp.config(state="normal"); v["steps"].set(s["steps"])
            else:
                stp.config(state="disabled")
            if s.get("guidance") is not None:
                gui.config(state="normal"); v["guidance"].set(s["guidance"])
            else:
                gui.config(state="disabled")
        cb.bind("<<ComboboxSelected>>", on_model)
        page.on_model = on_model

        prog = tk.Frame(page, bg=BG)
        prog.pack(fill="x", padx=16, pady=(6, 0))
        pb = ttk.Progressbar(prog, style="Bee.Horizontal.TProgressbar", maximum=100)
        pb.pack(side="left", fill="x", expand=True)
        BeeButton(prog, "Stop", self.cancel.set).pack(side="left", padx=(10, 0))

        view = tk.Frame(page, bg=BG)
        view.pack(fill="both", expand=True, padx=16, pady=10)
        canvas = tk.Canvas(view, bg=FIELD, highlightthickness=1, highlightbackground=BORDER)
        canvas.pack(side="left", fill="both", expand=True)
        strip = tk.Frame(view, bg=PANEL, width=150, highlightthickness=1, highlightbackground=BORDER)
        strip.pack(side="left", fill="y", padx=(10, 0))
        strip.pack_propagate(False)
        BeeButton(strip, "Save...", lambda: self._save_current(page)).pack(fill="x", padx=8, pady=(8, 4))
        BeeButton(strip, "-> Img2Img", lambda: self._send_i2i(page)).pack(fill="x", padx=8, pady=4)
        thumbs = tk.Frame(strip, bg=PANEL)
        thumbs.pack(fill="both", expand=True, pady=6)

        logw = tk.Text(page, height=4, bg=FIELD, fg=BLUE_LO, font=MONO, relief="flat",
                       highlightthickness=1, highlightbackground=BORDER, state="disabled")
        logw.pack(fill="x", padx=16, pady=(0, 12))

        page.v, page.prompt, page.gen, page.pb, page.canvas = v, prompt, gen, pb, canvas
        page.thumbs, page.logw, page.img2img = thumbs, logw, img2img
        page.shown, page.tk_img, page.init = None, None, None
        canvas.bind("<Configure>", lambda e: self._draw(page))

    def _pick_init(self, page):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if p:
            page.init = Image.open(p).convert("RGB")
            page.init_lbl.config(text=f"{os.path.basename(p)}  {page.init.size[0]}x{page.init.size[1]}", fg=BLUE)

    def _send_i2i(self, page):
        if page.shown is None:
            return
        tgt = self.pages["Image to Image"]
        tgt.init = page.shown.copy()
        tgt.init_lbl.config(text=f"from gallery  {tgt.init.size[0]}x{tgt.init.size[1]}", fg=BLUE)
        tgt.prompt.delete("1.0", "end"); tgt.prompt.insert("1.0", page.prompt.get("1.0", "end").strip())
        self.show("Image to Image")

    def _draw(self, page):
        c = page.canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 16), max(c.winfo_height(), 16)
        if page.shown is None:
            c.create_text(cw // 2, ch // 2, text="your images land here", fill=BLUE_LO, font=FONT_H)
            return
        im = page.shown.copy()
        im.thumbnail((max(cw - 8, 8), max(ch - 8, 8)), Image.LANCZOS)
        page.tk_img = ImageTk.PhotoImage(im)
        c.create_image(cw // 2, ch // 2, image=page.tk_img)

    def _add_thumb(self, page, img):
        t = img.copy(); t.thumbnail((128, 128))
        ti = ImageTk.PhotoImage(t)
        self.thumbs.append(ti)
        self.thumbs = self.thumbs[-40:]
        b = tk.Label(page.thumbs, image=ti, bg=PANEL, cursor="hand2",
                     highlightthickness=1, highlightbackground=BORDER)
        b.image = ti
        b.pack(pady=3)
        b.bind("<Button-1>", lambda e: (setattr(page, "shown", img), self._draw(page)))
        for k in page.thumbs.winfo_children()[:-6]:
            k.destroy()

    def _save_current(self, page):
        if page.shown is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            page.shown.save(p)
            self.log(f"saved {p}")

    def generate(self, page):
        if self.busy:
            return
        v = page.v
        prompt = page.prompt.get("1.0", "end").strip()
        if not prompt:
            return
        name = v["model"].get()
        spec = MODELS[name]
        try:
            w, h = (int(x) for x in v["size"].get().lower().replace(" ", "").split("x"))
            w, h = max(256, w // 16 * 16), max(256, h // 16 * 16)
        except Exception:
            messagebox.showerror(APP, "size must be WxH, e.g. 1024x1024"); return
        if page.img2img and page.init is None:
            messagebox.showerror(APP, "load an input image first"); return
        seed = num(v["seed"], int, -1)
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        params = dict(prompt=prompt, w=w, h=h,
                      steps=max(1, num(v["steps"], int, spec.get("steps") or 28)),
                      guidance=num(v["guidance"], float, spec.get("guidance") or 0.0),
                      seed=seed, count=max(1, min(8, num(v["count"], int, 1))),
                      strength=num(v["strength"], float, 0.75),
                      init_image=page.init if page.img2img else None,
                      offload=self.cfg.get("offload", True), hf_token=self.cfg.get("hf_token", ""),
                      bfl_key=self.cfg.get("bfl_key", ""), bfl_base=self.cfg.get("bfl_base", "https://api.bfl.ai"),
                      slug=self.cfg.get("flux3_slug", "flux-3") if spec.get("editable_slug") else None)
        self.busy = True
        self.cancel.clear()
        page.gen.set_enabled(False)
        page.pb["value"] = 0
        self._set_status(f"generating · {name}")
        self.log(f"{name} · {w}x{h} · seed {seed}")
        threading.Thread(target=self._worker, args=(page, name, spec, params), daemon=True).start()

    def _worker(self, page, name, spec, p):
        t0 = time.time()
        try:
            eng = self.api if spec["kind"] == "api" else self.local
            prog = lambda a, b: self.q.put(("prog", page, a / max(b, 1) * 100))
            stat = lambda text, frac: self.q.put(("dl", page, text, frac))
            res = eng.run(spec, p, prog, self.cancel, stat)
            for img, seed in res:
                fn = save_output(img, seed)
                meta = dict(file=fn, prompt=p["prompt"], model=name, seed=seed, w=p["w"], h=p["h"],
                            steps=p["steps"], guidance=p["guidance"], time=time.strftime("%Y-%m-%d %H:%M:%S"),
                            img2img=p["init_image"] is not None)
                self.q.put(("img", page, img, meta))
                self.q.put(("log", f"saved → {fn}"))
            self.q.put(("done", page, f"done · {len(res)} img · {time.time() - t0:.1f}s"))
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            self.q.put(("done", page, "error: " + friendly(e)))

    def _pump(self):
        try:
            while True:
                m = self.q.get_nowait()
                k = m[0]
                if k == "prog":
                    m[1].pb["value"] = m[2]
                elif k == "dl":
                    _, page, text, frac = m
                    page.pb["value"] = frac * 100
                    self._set_status(text)
                    now = time.time()
                    if frac >= 1.0 or now - getattr(self, "_dl_t", 0) > 5:   # log line every 5 s
                        self._dl_t = now
                        self._log_now(text)
                elif k == "img":
                    _, page, img, meta = m
                    page.shown = img
                    self._draw(page)
                    self._add_thumb(page, img)
                    self.history.insert(0, meta)
                    save_json(HIST, self.history[:500])
                elif k == "log":
                    self._log_now(m[1])
                elif k == "done":
                    _, page, msg = m
                    self.busy = False
                    page.gen.set_enabled(True)
                    page.pb["value"] = 0 if msg.startswith("error") else 100
                    self._set_status(msg)
                    self._log_now(msg)
        except queue.Empty:
            pass
        self.after(50, self._pump)

    def log(self, s):
        self.q.put(("log", s))

    def _log_now(self, s):
        print(f"[bee] {s.rstrip()}", flush=True)
        for name in ("Text to Image", "Image to Image"):
            w = self.pages[name].logw
            w.config(state="normal")
            w.insert("end", s.rstrip() + "\n")
            w.see("end")
            w.config(state="disabled")

    def _set_status(self, s):
        self.status.config(text=s)

    def _history_page(self, page):
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=16, pady=14)
        lbl(bar, "History", font=FONT_H, bg=BG).pack(side="left")
        BeeButton(bar, "Open folder", self._open_out).pack(side="right")
        BeeButton(bar, "Clear", self._clear_hist).pack(side="right", padx=8)
        wrap = tk.Frame(page, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        c = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=c.yview, bg=BTN, troughcolor=FIELD)
        c.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        c.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(c, bg=BG)
        c.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))

        def wheel(e):
            if self.current != "History":
                return
            if getattr(e, "num", None) in (4, 5):          # X11
                c.yview_scroll(-1 if e.num == 4 else 1, "units")
            elif e.delta:                                   # Win / macOS
                c.yview_scroll(-1 if e.delta > 0 else 1, "units")
        c.bind_all("<MouseWheel>", wheel)
        c.bind_all("<Button-4>", wheel)
        c.bind_all("<Button-5>", wheel)
        page.inner, page.hist_imgs = inner, []

    def _refresh_history(self):
        page = self.pages["History"]
        for w in page.inner.winfo_children():
            w.destroy()
        page.hist_imgs.clear()
        cols, i = 5, 0
        for h in self.history[:100]:
            if not os.path.exists(h.get("file", "")):
                continue
            try:
                im = Image.open(h["file"]); im.thumbnail((180, 180))
                ti = ImageTk.PhotoImage(im)
            except Exception:
                continue
            card = tk.Frame(page.inner, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=i // cols, column=i % cols, padx=6, pady=6, sticky="n")
            i += 1
            page.hist_imgs.append(ti)
            il = tk.Label(card, image=ti, bg=PANEL, cursor="hand2")
            il.pack(padx=4, pady=4)
            il.bind("<Button-1>", lambda e, hh=h: self._reuse(hh))
            lbl(card, h.get("prompt", "")[:60], wraplength=180, justify="left").pack(padx=4)
            lbl(card, f"{h.get('model', '?')}\nseed {h.get('seed')} · {h.get('w')}x{h.get('h')}", fg=BLUE_LO,
                font=(FONT[0], 8), justify="left").pack(padx=4, pady=(0, 4))

    def _reuse(self, h):
        name = "Image to Image" if h.get("img2img") else "Text to Image"
        page = self.pages[name]
        page.prompt.delete("1.0", "end"); page.prompt.insert("1.0", h.get("prompt", ""))
        if h.get("model") in MODELS:
            page.v["model"].set(h["model"]); page.on_model()
        page.v["seed"].set(str(h.get("seed", -1))); page.v["size"].set(f"{h.get('w', 1024)}x{h.get('h', 1024)}")
        page.v["steps"].set(h.get("steps", 28)); page.v["guidance"].set(h.get("guidance", 0.0))
        page.shown = Image.open(h["file"]).convert("RGB")
        self.show(name)
        self._draw(page)

    def _clear_hist(self):
        if messagebox.askyesno(APP, "clear history list? (files stay in outputs/)"):
            self.history = []; save_json(HIST, []); self._refresh_history()

    def _open_out(self):
        if sys.platform == "win32":
            os.startfile(OUT)
        elif sys.platform == "darwin":
            os.system(f'open "{OUT}"')
        else:
            os.system(f'xdg-open "{OUT}" &')

    def _settings_page(self, page):
        lbl(page, "Settings", font=FONT_H, bg=BG).pack(anchor="w", padx=16, pady=14)
        box = tk.Frame(page, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="x", padx=16)
        sv = dict(bfl_key=tk.StringVar(value=self.cfg.get("bfl_key", "")),
                  hf_token=tk.StringVar(value=self.cfg.get("hf_token", "")),
                  bfl_base=tk.StringVar(value=self.cfg.get("bfl_base", "https://api.bfl.ai")),
                  flux3_slug=tk.StringVar(value=self.cfg.get("flux3_slug", "flux-3")),
                  offload=tk.BooleanVar(value=self.cfg.get("offload", True)))
        rows = [("BFL API key", "bfl_key", "*"), ("HF token (optional)", "hf_token", "*"),
                ("BFL base URL", "bfl_base", None), ("FLUX.3 endpoint slug", "flux3_slug", None)]
        for i, (t, k, show) in enumerate(rows):
            lbl(box, t).grid(row=i, column=0, sticky="e", padx=10, pady=6)
            entry(box, sv[k], 48, show).grid(row=i, column=1, sticky="w", padx=10, pady=6)
        tk.Checkbutton(box, text="CPU offload (low VRAM)", variable=sv["offload"], bg=PANEL, fg=BLUE,
                       selectcolor=FIELD, activebackground=PANEL, activeforeground=BLUE_HI,
                       font=FONT).grid(row=len(rows), column=1, sticky="w", padx=10, pady=6)
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=16, pady=10)

        def save():
            for k, var in sv.items():
                self.cfg[k] = var.get()
            save_json(CFG, self.cfg)
            self._set_status("settings saved")
        BeeButton(bar, "Save", save).pack(side="left")
        BeeButton(bar, "Unload local model", lambda: (self.local.unload(), self._set_status("model unloaded"))
                  ).pack(side="left", padx=8)
        info = ("Local (no HF login): FLUX.1 schnell / dev, FLUX.2 klein 4B / dev — diffusers, CUDA/MPS/CPU.\n"
                "Cloud: FLUX.2 pro/max/flex/klein + FLUX.3 via api.bfl.ai (x-key header, polled).\n"
                f"Outputs: {OUT}")
        lbl(page, info, bg=BG, fg=BLUE_LO, justify="left").pack(anchor="w", padx=16, pady=8)

    def _quit(self):
        self.cancel.set()
        self.destroy()


def main():
    if len(sys.argv) > 1:
        sys.exit(cli(sys.argv[1:]))
    if tk is None:
        print("tkinter not available → use the CLI:  python diffusionbee.py -p \"a cat\" -m schnell")
        print("  (or install Tk: Debian/Ubuntu `sudo apt install python3-tk`, macOS `brew install python-tk`)")
        sys.exit(1)
    CatsDiffusionBee().mainloop()


if __name__ == "__main__":
    main()
