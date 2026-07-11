#!/usr/bin/env python
"""
Horizon Forecast — standalone nowcasting demo viewer.

Offline Tkinter+Pillow viewer over a pre-rendered asset bank (no torch/CUDA). Pick a case,
toggle Driver-First Cascade vs End-to-End Ablation (vs Compare), scrub the forecast horizon
T+15..T+120, switch Clouds&Rain vs Drivers (wind/temp). Shows prediction vs ground truth + metrics.

Build to .exe with build_demo_exe.bat (PyInstaller --onefile --windowed).
"""
import json, os, sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# locate bundled assets (works in dev + PyInstaller onefile)
BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(BASE, "assets")
MANIFEST = os.path.join(BASE, "manifest.json")

NAVY = "#0E2A47"; CYAN = "#2EC4D8"; BG = "#0d1117"; PANEL = "#16202b"; INK = "#e6edf3"; GREY = "#8b97a3"
HORIZONS = [15, 30, 45, 60, 75, 90, 105, 120]


class DemoApp:
    """Tkinter viewer: builds the window, wires the controls, and redraws panels on any change."""

    def __init__(self, root):
        """Load the manifest, init UI state (case/model/view/horizon), build the layout, first draw."""
        self.root = root
        self.m = json.load(open(MANIFEST, encoding="utf-8"))
        self.cases = self.m["cases"]
        self.mlabel = self.m["models"]
        self.cache = {}                      # (file,w,h) -> PhotoImage
        self.playing = False
        self.case_i = 0
        self.model = tk.StringVar(value="compare")
        self.view = tk.StringVar(value="clouds")
        self.hz = tk.IntVar(value=15)

        root.title("Horizon Forecast — Nowcasting Demo")
        root.configure(bg=BG)
        root.geometry("1340x910")
        root.minsize(1100, 740)
        self._build_header()
        body = tk.Frame(root, bg=BG); body.pack(fill="both", expand=True)
        self._build_controls(body)
        self.display = tk.Frame(body, bg=BG); self.display.pack(side="left", fill="both", expand=True)
        self._build_footer()
        self.redraw()

    # image loading
    def _img(self, fname, scale=1.0):
        """Load asset <fname> for the current case, resize by `scale`, cache and return a PhotoImage."""
        path = os.path.join(ASSETS, self.cases[self.case_i]["id"], fname)
        if not os.path.exists(path):
            return None
        im = Image.open(path)
        w, h = int(im.width * scale), int(im.height * scale)
        key = (path, w, h)
        if key not in self.cache:
            self.cache[key] = ImageTk.PhotoImage(im.resize((w, h), Image.LANCZOS))
        return self.cache[key]

    # header
    def _build_header(self):
        """Draw the top navy title bar (project name + tagline)."""
        h = tk.Frame(self.root, bg=NAVY, height=64); h.pack(fill="x"); h.pack_propagate(False)
        tk.Label(h, text="HORIZON FORECAST", bg=NAVY, fg="white",
                 font=("Segoe UI", 20, "bold")).pack(side="left", padx=18)
        tk.Label(h, text="Driver-First Cascaded Nowcasting  ·  satellite → wind/temp → clouds & rain",
                 bg=NAVY, fg=CYAN, font=("Segoe UI", 11)).pack(side="left", padx=4)

    def _build_footer(self):
        """Draw the bottom navy credits bar (authors, college, project code, test period)."""
        f = tk.Frame(self.root, bg=NAVY, height=30); f.pack(fill="x", side="bottom"); f.pack_propagate(False)
        tk.Label(f, text="Or Hod · Gilad Boudman   |   Braude College  ·  Project 26-1-R-1   |   held-out test set (2024-07 … 2025-12)",
                 bg=NAVY, fg=GREY, font=("Segoe UI", 9)).pack(side="left", padx=14)

    # controls
    def _build_controls(self, parent):
        """Build the left control panel: case picker, view/model radios, horizon slider, play button, caption."""
        c = tk.Frame(parent, bg=PANEL, width=280); c.pack(side="left", fill="y"); c.pack_propagate(False)
        def hdr(t): tk.Label(c, text=t, bg=PANEL, fg=CYAN, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(14, 2))

        hdr("CASE")
        self.case_names = [f"{i+1}. {cc['type']} — {cc['label']}" for i, cc in enumerate(self.cases)]
        self.case_var = tk.StringVar(value=self.case_names[0])
        om = ttk.Combobox(c, textvariable=self.case_var, values=self.case_names, state="readonly", width=34)
        om.pack(padx=14, fill="x"); om.bind("<<ComboboxSelected>>", self._on_case)

        hdr("VIEW")
        for val, txt in [("clouds", "Clouds & Rain"), ("drivers", "Drivers (wind / temp)")]:
            tk.Radiobutton(c, text=txt, variable=self.view, value=val, command=self.redraw,
                           bg=PANEL, fg=INK, selectcolor=NAVY, activebackground=PANEL,
                           activeforeground=CYAN, font=("Segoe UI", 10)).pack(anchor="w", padx=20)

        hdr("MODEL")
        for val, txt in [("cascade", "Driver-First Cascade"), ("ablation", "End-to-End (Ablation)"), ("compare", "Compare (side-by-side)")]:
            tk.Radiobutton(c, text=txt, variable=self.model, value=val, command=self.redraw,
                           bg=PANEL, fg=INK, selectcolor=NAVY, activebackground=PANEL,
                           activeforeground=CYAN, font=("Segoe UI", 10)).pack(anchor="w", padx=20)

        hdr("FORECAST HORIZON")
        self.hz_lbl = tk.Label(c, text="T+15 min", bg=PANEL, fg=INK, font=("Segoe UI", 11, "bold"))
        self.hz_lbl.pack(anchor="w", padx=14)
        s = tk.Scale(c, from_=15, to=120, resolution=15, orient="horizontal", variable=self.hz,
                     command=self._on_hz, bg=PANEL, fg=INK, troughcolor=NAVY, highlightthickness=0,
                     showvalue=False, length=240)
        s.pack(padx=14)
        self.play_btn = tk.Button(c, text="▶  Play rollout", command=self._toggle_play,
                                  bg=CYAN, fg=NAVY, font=("Segoe UI", 10, "bold"), relief="flat")
        self.play_btn.pack(padx=14, pady=8, fill="x")

        hdr("CASE INFO")
        self.cap = tk.Label(c, text="", bg=PANEL, fg=GREY, font=("Segoe UI", 9), wraplength=250, justify="left")
        self.cap.pack(anchor="w", padx=14, pady=(0, 8))

    # events
    def _on_case(self, *_):
        """Combobox handler: switch the active case index and redraw."""
        self.case_i = self.case_names.index(self.case_var.get()); self.redraw()

    def _on_hz(self, *_):
        """Slider handler: update the horizon label (T+N) and redraw at the new lead time."""
        self.hz_lbl.config(text=f"T+{self.hz.get()} min"); self.redraw()

    def _toggle_play(self):
        """Start/stop the autoplay rollout, flips the button label and kicks off _tick when playing."""
        self.playing = not self.playing
        self.play_btn.config(text="⏸  Pause" if self.playing else "▶  Play rollout")
        if self.playing:
            self._tick()

    def _tick(self):
        """Autoplay step: advance the horizon by 15 min (wrapping 120→15), redraw, re-arm after 900 ms."""
        if not self.playing:
            return
        nxt = self.hz.get() + 15
        if nxt > 120: nxt = 15
        self.hz.set(nxt); self._on_hz()
        self.root.after(900, self._tick)

    # drawing
    def _panel(self, parent, title, fname, scale=1.02, sub=None):
        """Build one titled image tile (title + image + optional metric caption), shows a placeholder if missing."""
        f = tk.Frame(parent, bg=BG)
        tk.Label(f, text=title, bg=BG, fg=CYAN, font=("Segoe UI", 10, "bold")).pack()
        img = self._img(fname, scale)
        if img:
            lbl = tk.Label(f, image=img, bg=BG); lbl.image = img; lbl.pack()
        else:
            tk.Label(f, text="(rendering…)", bg=PANEL, fg=GREY, width=22, height=14).pack()
        if sub:
            tk.Label(f, text=sub, bg=BG, fg=INK, font=("Consolas", 9)).pack()
        return f

    def redraw(self):
        """Rebuild the whole display for the current (case, view, model, horizon): input strip + prediction/GT panels."""
        for w in self.display.winfo_children():
            w.destroy()
        case = self.cases[self.case_i]
        lead = self.hz.get(); H = f"T{lead:03d}"
        self.cap.config(text=f"{case['time']}\n\n{case['caption']}")

        # input thumbnails — IR + WV both span all 4 history frames (model sees both each
        # timestep). DEM static. Single compact strip: IR×4 · WV×4 · DEM.
        inp = tk.Frame(self.display, bg=BG); inp.pack(fill="x", pady=(8, 2))
        tk.Label(inp, text="INPUT  (satellite IR + water-vapor, 4 frames each, + terrain)",
                 bg=BG, fg=GREY, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10)
        row = tk.Frame(inp, bg=BG); row.pack()
        strip = [("input_ir_tm45.png", "IR t-45"), ("input_ir_tm30.png", "IR t-30"),
                 ("input_ir_tm15.png", "IR t-15"), ("input_ir_t0.png", "IR t0"),
                 ("input_wv_tm45.png", "WV t-45"), ("input_wv_tm30.png", "WV t-30"),
                 ("input_wv_tm15.png", "WV t-15"), ("input_wv_t0.png", "WV t0"),
                 ("input_dem.png", "DEM")]
        for fn, t in strip:
            self._panel(row, t, fn, scale=0.34).pack(side="left", padx=2)

        # main panels
        main = tk.Frame(self.display, bg=BG); main.pack(pady=6)
        models = ["cascade", "ablation"] if self.model.get() == "compare" else [self.model.get()]

        if self.view.get() == "clouds":
            sc = 0.92 if len(models) == 1 else 0.78      # 3-up compare needs smaller panels to fit
            for m in models:
                met = case["metrics"][m][str(lead)]
                self._panel(main, self.mlabel[m].split(" (")[0], f"{m}_cloudrain_{H}.png", scale=sc,
                            sub=f"CSI {met['CSI']:.2f}   SSIM {met['SSIM']:.2f}").pack(side="left", padx=8)
            self._panel(main, "Ground Truth", f"gt_cloudrain_{H}.png", scale=sc).pack(side="left", padx=8)
        else:  # drivers
            grid = tk.Frame(main, bg=BG); grid.pack()
            wrow = tk.Frame(grid, bg=BG); wrow.pack()
            trow = tk.Frame(grid, bg=BG); trow.pack(pady=(6, 0))
            sc = 0.54 if len(models) == 1 else 0.45      # two stacked rows must fit the window
            for m in models:
                met = case["metrics"][m][str(lead)]
                self._panel(wrow, f"{self.mlabel[m].split(' (')[0]} wind", f"{m}_wind_{H}.png",
                            scale=sc, sub=f"RMSE {met['RMSE_wind']:.1f} m/s").pack(side="left", padx=6)
                self._panel(trow, f"{self.mlabel[m].split(' (')[0]} temp", f"{m}_temp_{H}.png",
                            scale=sc, sub=f"RMSE {met['RMSE_temp']:.1f} °C").pack(side="left", padx=6)
            self._panel(wrow, "GT wind", f"gt_wind_{H}.png", scale=sc).pack(side="left", padx=6)
            self._panel(trow, "GT temp", f"gt_temp_{H}.png", scale=sc).pack(side="left", padx=6)


if __name__ == "__main__":
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()
