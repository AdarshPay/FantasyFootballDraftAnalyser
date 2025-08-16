# live_assistant_gui.py
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import requests
import sv_ttk

BASE = "https://api.sleeper.app/v1"

try:
    import winsound
    def beep():
        try: winsound.MessageBeep()
        except: pass
except Exception:
    def beep(): pass

def fmt_one(val, places=1):
    try:
        if val is None:
            return "—"
        v = float(val)
        if np.isnan(v):
            return "—"
        return f"{v:.{places}f}"
    except Exception:
        return "—"

def get_json(url, retries=5, pause=0.35):
    for i in range(retries):
        r = requests.get(url, timeout=20)
        if r.ok:
            return r.json()
        time.sleep(pause * (1.5 ** i))
    r.raise_for_status()


def starters_per_team(lineup):
    from collections import Counter
    c = Counter()
    for slot in lineup:
        if slot in ("BN", "IR", "TAXI"):
            continue
        if slot in ("WRRB_FLEX",):
            c["RB"] += 0.5; c["WR"] += 0.5
        elif slot in ("FLEX",):
            c["RB"] += 0.4; c["WR"] += 0.5; c["TE"] += 0.1
        elif slot in ("REC_FLEX",):
            c["WR"] += 0.7; c["TE"] += 0.3
        elif slot in ("SUPER_FLEX",):
            c["QB"] += 1.0
        else:
            c[slot] += 1
    return c

class Tooltip:
    """Lightweight tooltip that follows the mouse over a widget."""
    def __init__(self, widget, text_fn):
        self.widget = widget
        self.text_fn = text_fn  # function(event) -> text
        self.tw = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<Motion>", self._motion, add="+")
    def _enter(self, e): self._motion(e)
    def _leave(self, e):
        if self.tw is not None:
            self.tw.destroy()
            self.tw = None
    def _motion(self, e):
        text = self.text_fn(e)
        if not text:
            self._leave(e)
            return
        if self.tw is None:
            self.tw = tk.Toplevel(self.widget)
            self.tw.wm_overrideredirect(True)
            try:
                self.tw.attributes("-topmost", True)
            except Exception:
                pass
            self.label = tk.Label(
                self.tw, text=text, justify="left",
                background="#111827", foreground="white",
                borderwidth=1, relief="solid",
                font=("Segoe UI", 8)
            )
            self.label.pack(ipadx=6, ipady=3)
        else:
            self.label.config(text=text)
        # position a little offset from cursor
        x = e.x_root + 12
        y = e.y_root + 8
        self.tw.wm_geometry(f"+{x}+{y}")

class SuggestionRow(ttk.Frame):
    def __init__(self, master, player, scale_min, scale_max, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.player = player

        # --------- Left text ----------
        name = f"{player.get('full_name','?')} ({player.get('position','?')})"
        adp_val = player.get("adp", None)
        adp_str = "—" if adp_val is None or (isinstance(adp_val, float) and np.isnan(adp_val)) else str(adp_val)
        sub = f"ADP {adp_str} | Live VORP {fmt_one(player.get('vorp_live'),1)} | Mean {fmt_one(player.get('season_mean'),1)}"
        ttk.Label(self, text=name, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text=sub,  foreground="#555").grid(row=1, column=0, sticky="w")

        # --------- Right mini chart ----------
        self.canvas_w = 360
        self.canvas_h = 60
        self.pad_l, self.pad_r = 12, 12
        self.band_y0, self.band_y1 = 18, 30  # p10–p90 band
        self.axis_y = 46

        self.scale_min = scale_min
        self.scale_max = max(scale_min + 1e-6, scale_max)  # avoid zero width
        self.p10  = self._float_or_none(player.get("p10"))
        self.p90  = self._float_or_none(player.get("p90"))
        self.mean = self._float_or_none(player.get("season_mean"))
        
                # --- volatility metrics for tooltip ---
        Z10 = 1.2815515655446004  # z for 10th/90th percentile in a Normal
        # try to read season std if present; else infer from p10/p90
        self.sigma = self._float_or_none(self.player.get("season_std"))
        if self.sigma is None and (self.p10 is not None) and (self.p90 is not None) and (self.p90 >= self.p10):
            self.sigma = (self.p90 - self.p10) / (2.0 * Z10)

        # coefficient of variation (as fraction); may be None if mean is 0/None
        if self.sigma is not None and self.mean not in (None, 0.0):
            self.cv = self.sigma / float(self.mean)
        else:
            self.cv = None

        # handy to show the literal band width too
        self.band_width = (self.p90 - self.p10) if (self.p10 is not None and self.p90 is not None) else None


        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, highlightthickness=0)
        self.canvas.grid(row=0, column=1, rowspan=2, padx=(12,0))
        self._draw_bar()

        # Tooltip: show p10/mean/p90, VORP, ADP, and the cursor value on the axis
        Tooltip(self.canvas, self._tooltip_text)

    # ---------- drawing ----------
    def _x(self, v: float) -> int:
        v = max(self.scale_min, min(self.scale_max, v))
        w = self.canvas_w
        return int(self.pad_l + (v - self.scale_min) / (self.scale_max - self.scale_min) * (w - self.pad_l - self.pad_r))

    def _float_or_none(self, x):
        try:
            v = float(x)
            return None if np.isnan(v) else v
        except Exception:
            return None

    def _draw_bar(self):
        c = self.canvas
        w, h = self.canvas_w, self.canvas_h
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill="#ffffff", outline="")

        # Axis with 5 ticks + labels (pts)
        c.create_line(self.pad_l, self.axis_y, w - self.pad_r, self.axis_y, fill="#CBD5E1")  # slate-300
        ticks = np.linspace(self.scale_min, self.scale_max, 5)
        for t in ticks:
            xt = self._x(t)
            c.create_line(xt, self.axis_y-4, xt, self.axis_y+4, fill="#94A3B8")  # slate-400
            label = f"{t:.0f}" if t >= 100 else f"{t:.1f}"
            c.create_text(xt, self.axis_y+12, text=f"{label} pts", fill="#64748B", font=("Segoe UI", 8), anchor="n")

        # p10–p90 band
        if self.p10 is not None and self.p90 is not None and self.p90 >= self.p10:
            c.create_rectangle(self._x(self.p10), self.band_y0, self._x(self.p90), self.band_y1,
                               fill="#DBEAFE", outline="#93C5FD")  # blue-100/300

        # mean tick + label
        if self.mean is not None:
            xm = self._x(self.mean)
            c.create_line(xm, self.band_y0-5, xm, self.band_y1+5, fill="#111827", width=2)  # gray-900
            c.create_text(xm, self.band_y0-6, text=f"mean {fmt_one(self.mean)}", fill="#111827",
                          font=("Segoe UI", 8, "bold"), anchor="s")

    # ---------- tooltip ----------
    def _tooltip_text(self, event):
        # map cursor x -> value on axis (clamped to axis span)
        x_min = self.pad_l
        x_max = self.canvas_w - self.pad_r
        x = min(max(event.x, x_min), x_max)
        frac = (x - x_min) / max(1, (x_max - x_min))
        cursor_val = self.scale_min + frac * (self.scale_max - self.scale_min)

        adp = self.player.get("adp", "—")
        vorp = fmt_one(self.player.get("vorp_live"), 1)
        p10  = fmt_one(self.p10, 1)
        mean = fmt_one(self.mean, 1)
        p90  = fmt_one(self.p90, 1)

        # volatility bits
        sigma = fmt_one(self.sigma, 1)
        cv_pct = fmt_one((self.cv * 100.0) if self.cv is not None else None, 1)
        band = fmt_one(self.band_width, 1)

        name = self.player.get("full_name", "?")
        pos  = self.player.get("position", "?")
        cur  = fmt_one(cursor_val, 1)

        return (
            f"{name} ({pos})\n"
            f"cursor ≈ {cur} pts\n"
            f"p10 {p10} | mean {mean} | p90 {p90}\n"
            f"σ {sigma} pts  |  CV {cv_pct}%  |  band {band} pts\n"
            f"Live VORP {vorp} | ADP {adp}"
        )

class App:
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Sleeper Live Draft Assistant (Projections + Live VORP)")
        self.root.geometry("1200x760")

        # --- theme toggle ---
        self.dark_mode = tk.BooleanVar(value=True)
        sv_ttk.set_theme("dark" if self.dark_mode.get() else "light")

        # --- core state ---
        self.pool = None          # ADP pool
        self.proj = None          # projections (season_mean, vorp, p10, p90)
        self.merged = None        # pool merged w/ projections
        self.teams = 10
        self.rounds = 15
        self.draft_id = args.draft_id
        self.username = args.username
        self.slot = args.slot
        self.my_roster_id = None
        self.on_clock = False
        self.running = False
        self.current_pick_no = 1
        self.rank_metric = tk.StringVar(value=args.rank_col)  # 'vorp' or 'season_mean'
        self.start_slots = {"QB":1, "RB":2, "WR":2, "TE":1, "K":1, "DEF":1}

        # strategy knobs
        self.need_weight           = tk.DoubleVar(value=1.25)
        self.early_qb_cap_round    = tk.IntVar(value=6)
        self.early_qb_cap_max      = tk.IntVar(value=1)
        self.early_te_cap_round    = tk.IntVar(value=6)
        self.early_te_cap_max      = tk.IntVar(value=1)
        self.defer_k_def_until     = tk.IntVar(value=12)
        self.adp_reach_tol         = tk.IntVar(value=18)
        self.adp_reach_slope       = tk.IntVar(value=12)   # (kept if you use it elsewhere)
        self.extra_pos_decay       = tk.DoubleVar(value=0.70)
        self.late_relax_round      = tk.IntVar(value=10)
        
        self.adp_blend_start = tk.IntVar(value=10)   # round when ADP starts contributing
        self.adp_blend_max   = tk.DoubleVar(value=0.70)

        self.lineup_slots = []        # league lineup positions (QB, RB, RB, WR, WR, TE, K, DEF)
        self.my_player_ids = set()
        # safe defaults for first UI tick
        self.picks = []
        self.my_roster_id = None
        self.teams = getattr(self, "teams", 10)

        # ---------- Top toolbar ----------
        top = ttk.Frame(self.root, padding=6)
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Draft ID").grid(row=0, column=0, sticky="e")
        self.ent_draft = ttk.Entry(top, width=26)
        self.ent_draft.insert(0, self.draft_id or "")
        self.ent_draft.grid(row=0, column=1, padx=4)

        ttk.Label(top, text="Username").grid(row=0, column=2, sticky="e")
        self.ent_user = ttk.Entry(top, width=18)
        if self.username: self.ent_user.insert(0, self.username)
        self.ent_user.grid(row=0, column=3, padx=4)

        ttk.Label(top, text="Slot").grid(row=0, column=4, sticky="e")
        self.ent_slot = ttk.Entry(top, width=6)
        if self.slot: self.ent_slot.insert(0, str(self.slot))
        self.ent_slot.grid(row=0, column=5, padx=4)

        ttk.Label(top, text="Pool CSV").grid(row=0, column=6, sticky="e")
        self.ent_csv = ttk.Entry(top, width=30)
        self.ent_csv.insert(0, args.csv)
        self.ent_csv.grid(row=0, column=7, padx=4)
        ttk.Button(top, text="…", command=self.pick_csv).grid(row=0, column=8)

        ttk.Label(top, text="Projections CSV").grid(row=1, column=0, sticky="e")
        self.ent_ranks = ttk.Entry(top, width=26)
        if args.ranks: self.ent_ranks.insert(0, args.ranks)
        self.ent_ranks.grid(row=1, column=1, padx=4, sticky="w")
        ttk.Button(top, text="…", command=self.pick_ranks).grid(row=1, column=2)

        ttk.Label(top, text="Rank by").grid(row=1, column=3, sticky="e")
        ttk.Combobox(top, textvariable=self.rank_metric,
                    values=["vorp","season_mean"], width=14, state="readonly").grid(row=1, column=4, sticky="w")

        self.btn_connect = ttk.Button(top, text="Connect", command=self.connect)
        self.btn_connect.grid(row=1, column=5, padx=4)

        self.status = ttk.Label(top, text="Idle")
        self.status.grid(row=1, column=6, columnspan=2, sticky="w")

        # Dark/Light toggle
        ttk.Checkbutton(
            top, text="Dark", variable=self.dark_mode,
            command=lambda: (sv_ttk.set_theme("dark" if self.dark_mode.get() else "light"),
                            self.refresh_ui_once())
        ).grid(row=0, column=9, padx=8, sticky="w")

        # ---------- Rebuild Projections panel ----------
        knobs = ttk.Labelframe(self.root, text="Rebuild Projections", padding=6)
        knobs.pack(side="top", fill="x", padx=6, pady=(2,6))

        self.var_prior = tk.IntVar(value=args.prior_games)
        self.var_weeks = tk.IntVar(value=args.weeks)
        self.var_sims  = tk.IntVar(value=args.sims)
        self.var_nudge = tk.DoubleVar(value=args.adp_nudge)

        ttk.Label(knobs, text="prior_games").grid(row=0, column=0, sticky="e")
        ttk.Spinbox(knobs, from_=0, to=20, textvariable=self.var_prior, width=6).grid(row=0, column=1)
        ttk.Label(knobs, text="weeks").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(knobs, from_=8, to=18, textvariable=self.var_weeks, width=6).grid(row=0, column=3)
        ttk.Label(knobs, text="sims").grid(row=0, column=4, sticky="e")
        ttk.Spinbox(knobs, from_=100, to=5000, increment=100, textvariable=self.var_sims, width=8).grid(row=0, column=5)
        ttk.Label(knobs, text="ADP nudge %").grid(row=0, column=6, sticky="e")
        ttk.Spinbox(knobs, from_=0, to=30, textvariable=self.var_nudge, width=6).grid(row=0, column=7)
        ttk.Button(knobs, text="Rebuild now", command=self.rebuild_projections).grid(row=0, column=8, padx=8)

        # ---------- Main area ----------
        main = ttk.Frame(self.root)
        # Pack the main area to the LEFT so the right sidebar can sit on the right
        main.pack(side="left", fill="both", expand=True)

        # Left: Suggestions list with scroll
        left = ttk.Frame(main, padding=(6,6))
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Suggestions (live VORP with risk)", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.sugg_canvas = tk.Canvas(left, borderwidth=0, highlightthickness=0)
        self.sugg_frame = ttk.Frame(self.sugg_canvas)
        self.sugg_scroll = ttk.Scrollbar(left, orient="vertical", command=self.sugg_canvas.yview)
        self.sugg_canvas.configure(yscrollcommand=self.sugg_scroll.set)
        self.sugg_scroll.pack(side="right", fill="y")
        self.sugg_canvas.pack(side="left", fill="both", expand=True)
        self.sugg_canvas.create_window((0,0), window=self.sugg_frame, anchor="nw")
        self.sugg_frame.bind("<Configure>", lambda e: self.sugg_canvas.configure(scrollregion=self.sugg_canvas.bbox("all")))

        # Center: counts + recent picks
        center = ttk.Frame(main, padding=(6,6))
        center.pack(side="left", fill="y")
        ttk.Label(center, text="Your Roster Counts", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_roster = ttk.Label(center, text="", justify="left")
        self.lbl_roster.pack(anchor="w", pady=(0,6))

        ttk.Label(center, text="Recent Picks", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tv = ttk.Treeview(center, columns=("team","pos","adp"), show="headings", height=15)
        for c,w in (("team",180),("pos",60),("adp",80)):
            self.tv.heading(c, text=c.upper())
            self.tv.column(c, width=w, anchor="center")
        self.tv.pack(fill="y", expand=False)

        # ---------- Right sidebar: Strategy (top) + My Roster (bottom) ----------
        self.sidebar = ttk.Frame(self.root)
        self.sidebar.pack(side="right", fill="y", padx=12, pady=8)

        # Strategy panel
        self.side = ttk.Labelframe(self.sidebar, text="Strategy")
        self.side.pack(side="top", fill="x", padx=0, pady=(0, 8))
        side = self.side
        
        row = 0      
        ttk.Label(side, text="ADP blend start").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=6, to=15, textvariable=self.adp_blend_start, width=6).grid(row=row, column=1); row+=1

        ttk.Label(side, text="ADP blend max").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=0.0, to=1.0, increment=0.05, textvariable=self.adp_blend_max, width=6).grid(row=row, column=1); row+=1

        ttk.Label(side, text="Need weight").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=1.0, to=2.0, increment=0.05,
                    textvariable=self.need_weight, width=6).grid(row=row, column=1); row+=1

        ttk.Label(side, text="QB cap (≤ round)").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=2, to=10, textvariable=self.early_qb_cap_round, width=6).grid(row=row, column=1)
        ttk.Label(side, text="max").grid(row=row, column=2, sticky="e")
        ttk.Spinbox(side, from_=0, to=2, textvariable=self.early_qb_cap_max, width=4).grid(row=row, column=3); row+=1

        ttk.Label(side, text="TE cap (≤ round)").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=2, to=10, textvariable=self.early_te_cap_round, width=6).grid(row=row, column=1)
        ttk.Label(side, text="max").grid(row=row, column=2, sticky="e")
        ttk.Spinbox(side, from_=0, to=2, textvariable=self.early_te_cap_max, width=4).grid(row=row, column=3); row+=1

        ttk.Label(side, text="Defer K/DEF until round").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=8, to=15, textvariable=self.defer_k_def_until, width=6).grid(row=row, column=1); row+=1

        ttk.Label(side, text="ADP reach tol (picks)").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=0, to=36, textvariable=self.adp_reach_tol, width=6).grid(row=row, column=1); row+=1

        ttk.Label(side, text="Early-caps relax after round").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(side, from_=6, to=15, textvariable=self.late_relax_round, width=6).grid(row=row, column=1); row+=1

        # My Roster panel under Strategy
        self.roster_frame = ttk.Labelframe(self.sidebar, text="My Roster")
        self.roster_frame.pack(side="top", fill="both", expand=True, padx=0, pady=0)

        self.roster_counts = ttk.Label(self.roster_frame, text="—")
        self.roster_counts.pack(anchor="w", padx=8, pady=(6, 4))

        cols = ("slot", "player", "pos", "adp", "mean", "vorp")
        self.roster_tree = ttk.Treeview(self.roster_frame, columns=cols, show="headings", height=14)
        for c, w in (("slot", 90), ("player", 180), ("pos", 44), ("adp", 60), ("mean", 70), ("vorp", 70)):
            self.roster_tree.heading(c, text=c.upper())
            self.roster_tree.column(c, width=w, anchor="w")
        self.roster_tree.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # ---------- Load data & kick UI ----------
        self.load_pool(args.csv)
        if args.ranks and os.path.exists(args.ranks):
            self.load_projections(args.ranks)

        # first refresh tick
        self.root.after(300, self.refresh_ui_once)
        
    def impute_mean_from_adp(self, df: pd.DataFrame) -> pd.Series:
        # Returns a Series of estimated means (index aligned to df)
        est = pd.to_numeric(df.get("season_mean"), errors="coerce").copy()
        adp = pd.to_numeric(df.get("adp"), errors="coerce")
        for pos, grp_idx in df.groupby("position").groups.items():
            idx = pd.Index(grp_idx)
            have = idx[df.loc[idx, "season_mean"].notna() & adp.loc[idx].notna()]
            miss = idx[df.loc[idx, "season_mean"].isna() & adp.loc[idx].notna()]
            if len(have) >= 10 and len(miss) > 0:
                x = adp.loc[have].values.astype(float)
                y = df.loc[have, "season_mean"].values.astype(float)
                try:
                    # simple robust-ish fit
                    b1, b0 = np.polyfit(x, y, 1)
                    est.loc[miss] = b0 + b1 * adp.loc[miss].values
                except Exception:
                    pass
        return est

        
    def _debug(self, msg: str):
        if getattr(self.args, "debug", False):
            try:
                print(f"[DEBUG] {msg}")
            except Exception:
                pass

    def _fetch_participants(self):
        """Load draft participants and keep quick lookups by user_id and roster_id."""
        try:
            parts = get_json(f"{BASE}/draft/{self.draft_id}/participants")  # works for mock drafts
        except Exception:
            parts = []
        self.participants = parts or []
        # maps
        self.user_by_display = { (p.get("display_name") or "").lower(): p for p in self.participants }
        self.part_by_uid = { str(p.get("user_id")): p for p in self.participants }
        self.part_by_rid = { str(p.get("roster_id")): p for p in self.participants }

    def _ensure_my_identity(self):
        """
        Determine my user_id and roster_id.
        Priority:
        - username (display_name) match
        - slot match
        - infer from first pick we make (picked_by -> roster_id)
        """
        if not hasattr(self, "participants"):
            self._fetch_participants()

        # 1) by username (display_name)
        if self.username:
            me = self.user_by_display.get(self.username.lower())
            if me:
                self.my_user_id = str(me.get("user_id") or "")
                self.my_roster_id = str(me.get("roster_id") or self.my_roster_id or "")
        # 2) by slot (participants often have 'slot' on mocks)
        if (not self.my_roster_id) and self.slot is not None:
            for p in self.participants:
                try:
                    if int(p.get("slot", -1)) == int(self.slot):
                        self.my_user_id = str(p.get("user_id") or "")
                        self.my_roster_id = str(p.get("roster_id") or "")
                        break
                except Exception:
                    pass

        # 3) infer from picks we made (if we know user_id but not roster_id)
        if (not self.my_roster_id) and getattr(self, "picks", None):
            for pick in self.picks:
                uid = str(pick.get("picked_by") or "")
                if self.username and uid and uid in self.part_by_uid:
                    who = self.part_by_uid[uid]
                    # only claim this if display name matches our username
                    if (who.get("display_name") or "").lower() == self.username.lower():
                        self.my_user_id = uid
                        self.my_roster_id = str(pick.get("roster_id") or who.get("roster_id") or "")
                        break

    def _update_taken_and_my_roster(self):
        """Mark taken players and build my roster rows from picks (robust for mocks)."""
        # --- 1) Taken set ---
        taken_ids = set()
        for p in (self.picks or []):
            pid = str(p.get("player_id") or "")
            if pid:
                taken_ids.add(pid)

        if self.merged is not None and "player_id" in self.merged.columns:
            self.merged["player_id"] = self.merged["player_id"].astype(str)
            self.merged["taken"] = self.merged["player_id"].isin(taken_ids)

        # --- 2) Figure out which picks are mine ---
        my_ids = set()

        # (a) Try user_id/roster_id if available
        my_uid = str(getattr(self, "my_user_id", "") or "")
        my_rid = str(getattr(self, "my_roster_id", "") or "")

        for p in (self.picks or []):
            pid = str(p.get("player_id") or "")
            if not pid:
                continue
            rid = str(p.get("roster_id") or "")
            uid = str(p.get("picked_by") or "")
            if (my_rid and rid and rid == my_rid) or (my_uid and uid and uid == my_uid):
                my_ids.add(pid)

        # (b) Fallback by pick number → slot (works in mocks where roster_id is None)
        try:
            my_slot = int(self.ent_slot.get() or self.slot or 1)
        except Exception:
            my_slot = 1
        teams = int(getattr(self, "teams", 10) or 10)

        for p in (self.picks or []):
            pid = str(p.get("player_id") or "")
            if not pid:
                continue
            # Sleeper uses "pick_no" on picks; some dumps also use "pick"
            pn = p.get("pick_no", p.get("pick"))
            try:
                pn = int(pn or 0)
            except Exception:
                pn = 0
            if pn > 0 and self._slot_for_pick_no(pn, teams) == my_slot:
                my_ids.add(pid)

        # --- 3) Commit and render ---
        self.my_player_ids = set(my_ids)
        self.refresh_roster_panel()


    def refresh_roster_panel(self, rep_points=None):
        """Render counts + table on the right sidebar."""
        # wipe tree
        for n in self.roster_tree.get_children():
            self.roster_tree.delete(n)

        if self.merged is None or not self.lineup_slots:
            self.roster_counts.config(text="QB 0/1 · RB 0/2 · WR 0/2 · TE 0/1 · K 0/1 · DEF 0/1")
            return

        df = self.merged.copy()
        df["player_id"] = df["player_id"].astype(str)
        
        ids = {str(x) for x in self.my_player_ids}
        mine = df[df["player_id"].astype(str).isin(ids)].copy()

        # counts
        need = self.start_slots
        counts = {k: 0 for k in need.keys()}
        for pos, c in mine["position"].value_counts().items():
            counts[pos] = int(c)

        counts_str = " · ".join([f"{pos} {counts.get(pos,0)}/{need[pos]}" for pos in ["QB","RB","WR","TE","K","DEF"]])
        self.roster_counts.config(text=counts_str)

        # show rows
        def fmt(x):
            try:
                if pd.isna(x): return ""
                return f"{float(x):.1f}"
            except Exception:
                return str(x) if x is not None else ""

        display_cols = ["full_name","position","adp","season_mean"]
        # choose live vorp if exists
        vcol = "vorp_live" if "vorp_live" in mine.columns else ("vorp" if "vorp" in mine.columns else None)

        # make a pleasant order: QB,RB,WR,TE,K,DEF
        order_key = {"QB":0,"RB":1,"WR":2,"TE":3,"K":4,"DEF":5}
        mine["__ord"] = mine["position"].map(order_key).fillna(9)
        mine = mine.sort_values(["__ord","adp"], na_position="last")

        slot_num = 1
        for _, r in mine.iterrows():
            self.roster_tree.insert(
                "", "end",
                values=(
                    f"#{slot_num}",
                    r.get("full_name",""),
                    r.get("position",""),
                    fmt(r.get("adp","")),
                    fmt(r.get("season_mean","")),
                    fmt(r.get(vcol,"")) if vcol else ""
                )
            )
            slot_num += 1
            
    def _slot_for_pick_no(self, pick_no: int, teams: int) -> int:
        """Return draft slot (1..teams) that owns a given global pick number in a snake draft."""
        if not pick_no or not teams:
            return -1
        rnd = (pick_no - 1) // teams + 1
        idx = (pick_no - 1) % teams + 1
        return idx if (rnd % 2 == 1) else (teams - idx + 1)
        
    def _id_to_pos_map(self):
        # build once at load; call where you load self.pool
        if not hasattr(self, "_id2pos"):
            self._id2pos = {}
            for _, r in self.pool.iterrows():
                self._id2pos[str(r.get("player_id"))] = str(r.get("position"))
        return self._id2pos

    def _current_round(self) -> int:
        try:
            teams = int(getattr(self, "teams", 10) or 10)
            pick_no = int(getattr(self, "current_pick_no", 1) or 1)
            return (pick_no - 1) // teams + 1
        except Exception:
            return 1

    def my_pos_counts(self) -> dict:
        """Return counts of my roster by POS (QB/RB/WR/TE/K/DEF)."""
        if self.merged is None:
            return {}
        ids = {str(x) for x in getattr(self, "my_player_ids", set())}
        if not ids:
            return {}
        df = self.merged[self.merged["player_id"].astype(str).isin(ids)]
        return df["position"].value_counts().to_dict()

    def _need_multiplier(self, pos: str, counts: dict, round_no: int) -> float:
        """How much to up/down weight a position given my roster + round."""
        start = self.start_slots  # e.g. {"QB":1,"RB":2,"WR":2,"TE":1,"K":1,"DEF":1}
        need = max(0, int(start.get(pos, 0)) - int(counts.get(pos, 0)))

        # Defer K/DEF completely until the configured round
        if pos in ("K", "DEF") and round_no < int(self.defer_k_def_until.get()):
            return 0.01

        # Early caps for QB/TE (hide extras before configured round)
        if pos == "QB" and round_no <= int(self.early_qb_cap_round.get()) and counts.get("QB", 0) >= int(self.early_qb_cap_max.get()):
            return 0.001
        if pos == "TE" and round_no <= int(self.early_te_cap_round.get()) and counts.get("TE", 0) >= int(self.early_te_cap_max.get()):
            return 0.001

        # If we still owe starters at *any* position and this pos is already satisfied,
        # strongly down-weight until late_relax_round.
        if need == 0:
            if round_no < int(self.late_relax_round.get()):
                if any(start[p] > counts.get(p, 0) for p in start.keys()):
                    return 0.25   # show but deprioritize
            # Otherwise use the “extra position” decay (diminishing returns)
            return float(self.extra_pos_decay.get())

        # We need this position — boost it by need_weight per missing slot
        return 1.0 + float(self.need_weight.get()) * need

    def _adp_reach_factor(self, adp: float, current_pick: int) -> float:
        """Soft penalty for reaching way past ADP early."""
        try:
            adp = float(adp)
        except Exception:
            return 1.0
        tol = int(self.adp_reach_tol.get())
        slope = max(1, int(self.adp_reach_slope.get()))
        diff = adp - float(current_pick)
        if diff <= tol:
            return 1.0
        # smooth exponential falloff
        import math
        return math.exp(-(diff - tol) / float(slope))

    def rank_candidates(self, topk: int = 12):
        """
        Roster-aware ranking with:
        - K/DEF deferral
        - QB/TE early caps
        - Need multiplier & ADP reach penalty
        - ADP blending that increases in later rounds (or 100% if projection missing)
        Also computes vorp_live for display; if 'season_mean' missing, tries a light ADP-based imputation for labels.
        """
        df = (self.merged if self.merged is not None else self.pool)
        if df is None or df.empty: return df
        df = df.copy()

        # Hide taken players if column exists
        if "taken" in df.columns:
            df = df[~df["taken"]]

        # Numeric fields
        metric = self.rank_metric.get()
        if metric not in df.columns:
            metric = "vorp" if "vorp" in df.columns else "season_mean"
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df["adp"]   = pd.to_numeric(df.get("adp"), errors="coerce")

        # Draft context
        round_no     = self._current_round()
        counts       = self.my_pos_counts() or {}
        current_pick = int(getattr(self, "current_pick_no", 1) or 1)

        # LIVE replacement on available pool for display VORP
        base_for_repl = "season_mean"
        df[base_for_repl] = pd.to_numeric(df.get(base_for_repl), errors="coerce")
        # (Optional) impute missing mean from ADP for *display* only
        try:
            missing_mask = df[base_for_repl].isna()
            if missing_mask.any():
                df.loc[missing_mask, base_for_repl] = self.impute_mean_from_adp(df).loc[missing_mask]
        except Exception:
            pass

        repl = self.compute_live_replacement(df)  # expects 'position' & 'season_mean'
        df["vorp_live"] = df[base_for_repl] - df["position"].map(repl).astype(float)

        # --- HARD FILTERS ---
        # 1) Defer K/DEF
        before = len(df)
        if round_no < int(self.defer_k_def_until.get()):
            df = df[~df["position"].isin(["K", "DEF"])]
        removed_kdef = before - len(df)

        # 2) Early caps for QB/TE
        before = len(df)
        if round_no <= int(self.early_qb_cap_round.get()) and counts.get("QB", 0) >= int(self.early_qb_cap_max.get()):
            df = df[df["position"] != "QB"]
        if round_no <= int(self.early_te_cap_round.get()) and counts.get("TE", 0) >= int(self.early_te_cap_max.get()):
            df = df[df["position"] != "TE"]
        removed_caps = before - len(df)

        # 3) Hide surplus QB/TE until relax round if any starters still missing
        missing_any = any(self.start_slots[p] > counts.get(p, 0) for p in self.start_slots)
        kept_elites = []
        if missing_any and round_no < int(self.late_relax_round.get()):
            elite_qb = df[df["position"] == "QB"].nlargest(1, metric)
            elite_te = df[df["position"] == "TE"].nlargest(1, metric)
            kept_elites = [elite_qb.get("full_name").values.tolist(), elite_te.get("full_name").values.tolist()]
            df = pd.concat([df[~df["position"].isin(["QB","TE"])], elite_qb, elite_te]).drop_duplicates()

        if df.empty:
            self._debug(f"round={round_no} pick={current_pick} counts={counts} -> no candidates after filters")
            return df

        # --- ADP BLEND (late rounds) ---
        # Normalized production metric
        prod = pd.to_numeric(df[metric], errors="coerce")
        if np.isfinite(prod.min(skipna=True)) and np.isfinite(prod.max(skipna=True)) and prod.max(skipna=True) > prod.min(skipna=True):
            prod_norm = (prod - prod.min(skipna=True)) / (prod.max(skipna=True) - prod.min(skipna=True))
        else:
            prod_norm = pd.Series(0.0, index=df.index)

        # Normalized "inverse ADP" (lower ADP -> higher score)
        adp_rank = df["adp"].rank(pct=True, method="max")
        adp_inv  = (1.0 - adp_rank).fillna(0.0)

        # Round-dependent ADP weight
        start  = int(self.adp_blend_start.get())
        wmax   = float(self.adp_blend_max.get())
        if round_no <= start:
            w_global = 0.0
        else:
            denom = max(1, self.rounds - start)
            w_global = min(wmax, wmax * (round_no - start) / denom)

        # Row-wise weight: if production missing on that player, force ADP weight to 1.0
        w_row = np.where(prod.isna(), 1.0, w_global)
        df["_base"] = (1.0 - w_row) * prod_norm.fillna(0.0) + w_row * adp_inv

        # --- SOFT SCORING ---
        need_weight     = float(self.need_weight.get())
        extra_pos_decay = float(self.extra_pos_decay.get())
        tol             = int(self.adp_reach_tol.get())
        slope           = max(1, int(self.adp_reach_slope.get()))

        starter_need = {p: max(0, int(self.start_slots.get(p, 0)) - int(counts.get(p, 0))) for p in self.start_slots}
        df["_need_mul"] = 1.0
        for pos in df["position"].unique():
            need = starter_need.get(pos, 0)
            df.loc[df["position"] == pos, "_need_mul"] = (1.0 + need_weight * need) if need > 0 else extra_pos_decay

        # ADP reach penalty
        diff = df["adp"] - float(current_pick)
        df["_reach_mul"] = np.where(diff <= tol, 1.0, np.exp(-(diff - tol) / float(slope)))

        # Final score
        df["score"] = df["_base"] * df["_need_mul"] * df["_reach_mul"]

        # cv% for tooltip if available
        if "season_std" in df.columns and "season_mean" in df.columns:
            s = pd.to_numeric(df["season_std"], errors="coerce")
            m = pd.to_numeric(df["season_mean"], errors="coerce").replace(0, np.nan)
            df["cv_pct"] = (s / m * 100.0).round(1).fillna(np.inf)

        top = df.sort_values("score", ascending=False).head(topk)

        # Debug snapshot
        self._debug(
            f"round={round_no} pick={current_pick} counts={counts} | defer_removed={removed_kdef} "
            f"cap_removed={removed_caps} | w_adp={w_global:.2f}"
        )

        return top



    # ---------- data loading ----------
    def load_pool(self, path):
        df = pd.read_csv(path)
        # expected: player_id, full_name, position, adp (team optional)
        if "team" not in df.columns: df["team"] = ""
        df["position"] = df["position"].str.upper().replace({"DST":"DEF","D/ST":"DEF"})
        df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
        self.pool = df

    def load_projections(self, path):
        p = pd.read_csv(path)
        keep = {"player_id","season_mean","season_std","p10","p90","vorp"}
        cols = [c for c in p.columns if c in keep]
        p = p[cols].copy()
        # merge into pool
        self.merged = self.pool.merge(p, on="player_id", how="left")
        # Fallback score if projection missing: use inverse ADP
        if self.args.rank_col == "vorp":
            self.merged["rank_score_base"] = self.merged["vorp"].fillna(-1e9)
        else:
            self.merged["rank_score_base"] = self.merged["season_mean"].fillna(-1e9)

    # ---------- connect / polling ----------
    def connect(self):
        self.draft_id = self.ent_draft.get().strip()
        self.username = self.ent_user.get().strip() or None
        self.slot = int(self.ent_slot.get()) if self.ent_slot.get().strip() else None

        if not self.draft_id:
            messagebox.showerror("Missing", "Enter a Draft ID")
            return

        draft = get_json(f"{BASE}/draft/{self.draft_id}")
        self.rounds = int(draft.get("settings", {}).get("rounds", 15))
        total_teams = int(draft.get("metadata", {}).get("teams") or draft.get("total_teams", 10))
        league_id = draft.get("league_id")
        self.teams = total_teams

        # Lineup
        if league_id:
            league = get_json(f"{BASE}/league/{league_id}")
            self.lineup_slots = league.get("roster_positions", [])
        else:
            # fallback if mock draft or no league
            self.lineup_slots = ["QB","RB","RB","WR","WR","TE","K","DEF"]

        # Resolve my_roster_id
        self.my_roster_id = None

        # If there IS a league_id, we can resolve via slot or username
        if league_id and self.slot:
            rosters = get_json(f"{BASE}/league/{league_id}/rosters")
            for r in rosters:
                rid = int(r.get("roster_id"))
                slot = int(r.get("slot", rid))
                if slot == int(self.slot) or rid == int(self.slot):
                    self.my_roster_id = rid
                    break
        elif league_id and self.username:
            if self.my_roster_id is None and self.username:
                users = get_json(f"{BASE}/league/{league_id}/users")
            uid = None
            for u in users:
                if str(u.get("display_name","")).lower() == self.username.lower():
                    uid = u.get("user_id"); break
            if uid:
                rosters = get_json(f"{BASE}/league/{league_id}/rosters")
                for r in rosters:
                    if r.get("owner_id") == uid:
                        self.my_roster_id = int(r["roster_id"]); break

        # If NO league_id (mock drafts), allow slot as a direct hint
        if self.my_roster_id is None and not league_id and self.slot:
            # In drafts without league context, roster_id is often 1..teams; assume slot==roster_id
            self.my_roster_id = int(self.slot)

        self.status.config(text=f"✅ Connected: teams={self.teams}, rounds={self.rounds}, my_roster_id={self.my_roster_id}")
        self.running = True
        threading.Thread(target=self.poll_loop, daemon=True).start()

    def poll_loop(self):
        recent = []
        while self.running:
            try:
                picks = get_json(f"{BASE}/draft/{self.draft_id}/picks")
                # sort by round, pick_no
                picks = sorted(picks, key=lambda x: (int(x.get("round", 0)), int(x.get("pick_no", 0))))
                # recent feed
                recent = [p for p in picks if p.get("player_id")]  # only made picks
                self.picks = recent
                # who is OTC?
                made = len(recent)
                self.current_pick_no = made + 1
                otc = None
                if made < len(picks):
                    otc = picks[made].get("roster_id")
                self.on_clock = (self.my_roster_id is not None and otc == self.my_roster_id) or self._my_pick_detect(recent)

                # update UI
                drafted_ids = set(str(p["player_id"]) for p in recent if p.get("player_id"))
                self.root.after(0, self._ensure_my_identity)
                self.root.after(0, self._update_taken_and_my_roster)
                picked_by = {str(p.get("player_id")): p.get("picked_by") for p in recent if p.get("player_id")}
                self.root.after(0, lambda: self.update_ui(recent[-20:], drafted_ids, picked_by))
                time.sleep(self.args.poll)
            except Exception as e:
                self.root.after(0, lambda: self.status.config(text=f"Error: {e}"))
                time.sleep(max(2.0, self.args.poll))

    def _my_pick_detect(self, recent):
        # Some events have roster_id None; also match on picked_by username if provided
        if not self.username:
            return False
        if not recent: return False
        return False  # keep simple—roster_id check in poll_loop is primary

    # ---------- live vorp & suggestions ----------
    def compute_live_replacement(self, avail):
        # starters across league from lineup
        sp_team = starters_per_team(self.lineup_slots)
        starters_league = {pos: int(round(sp_team[pos] * self.teams)) for pos in ("QB","RB","WR","TE","K","DEF")}
        repl = {}
        # choose base metric column
        base_col = "season_mean" if self.rank_metric.get() == "season_mean" else "season_mean"
        for pos in starters_league:
            pool_pos = avail[avail["position"] == pos].sort_values(base_col, ascending=False)
            taken_pos = self._taken_counts_by_pos().get(pos, 0)
            target_rank = max(starters_league[pos] - taken_pos, 1)
            idx = min(max(target_rank - 1, 0), max(len(pool_pos) - 1, 0))
            repl[pos] = float(pool_pos.iloc[idx][base_col]) if len(pool_pos) else 0.0
        return repl

    def _taken_counts_by_pos(self):
        # Quick estimation from recent picks table contents (pos in column 2)
        counts = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0}
        for iid in self.tv.get_children():
            vals = self.tv.item(iid, "values")
            if not vals: continue
            pos = vals[1]
            if pos in counts: counts[pos] += 1
        return counts

    def suggest(self, drafted_ids, topk=10):
        if self.merged is None:
            # fall back to ADP-only
            avail = self.pool[~self.pool["player_id"].astype(str).isin(drafted_ids)].copy()
            avail["season_mean"] = np.nan
            avail["p10"] = np.nan
            avail["p90"] = np.nan
        else:
            avail = self.merged[~self.merged["player_id"].astype(str).isin(drafted_ids)].copy()

        # live replacement thresholds (per position) based on remaining pool
        repl = self.compute_live_replacement(avail)

        # compute live VORP against those thresholds
        base = "season_mean" if self.rank_metric.get() == "season_mean" else "season_mean"
        avail["vorp_live"] = avail.apply(lambda r: (r.get(base) or -1e9) - repl.get(r["position"], 0.0), axis=1)

        # ADP proximity boost (closer to pick -> small bump)
        pick_idx = self.current_pick_no
        avail["adp_gap"] = -abs(avail["adp"] - pick_idx)
        # overall score: live vorp primary, ADP gap secondary
        avail["score"] = avail["vorp_live"].fillna(-1e9) * 1.0 + avail["adp_gap"].fillna(-999) * 0.02

        # Late K/DEF timing if your league uses them
        needs_k = "K" in self.lineup_slots
        needs_def = "DEF" in self.lineup_slots
        late_round = self.current_pick_no > (self.rounds - 2) * self.teams
        if (needs_k or needs_def) and late_round:
            # Small positive bias for K/DEF if not many left
            for p in ("K","DEF"):
                if p in self.lineup_slots:
                    left = (avail["position"] == p).sum()
                    if left > 0:
                        avail.loc[avail["position"] == p, "score"] += 3.0

        # Take topk by score
        top = avail.sort_values(["score","adp"], ascending=[False, True]).head(topk).copy()
        return top

    # ---------- UI update ----------
    def update_ui(self, recent_picks, drafted_ids, picked_by):
        # recent picks table
        for iid in self.tv.get_children(): self.tv.delete(iid)
        for p in reversed(recent_picks):
            pid = str(p.get("player_id"))
            meta = self.pool[self.pool["player_id"].astype(str) == pid]
            name = meta["full_name"].values[0] if len(meta) else pid
            pos  = meta["position"].values[0] if len(meta) else "?"
            adp  = meta["adp"].values[0] if len(meta) else "?"
            self.tv.insert("", "end", values=(name, pos, adp))

        # mark taken in merged so rank_candidates can filter
        if self.merged is not None:
            self.merged["player_id"] = self.merged["player_id"].astype(str)
            self.merged["taken"] = self.merged["player_id"].isin(drafted_ids)

        # compute current pick index
        self.current_pick_no = len(recent_picks) + 1

        # roster panel update (also updates counts used by ranker)
        self._ensure_my_identity()
        self._update_taken_and_my_roster()

        # suggestions via roster-aware ranker
        top = self.rank_candidates(topk=self.args.topk)
        for child in self.sugg_frame.winfo_children(): child.destroy()

        # scale for risk bars
        if top is not None and not top.empty and top["p10"].notna().any():
            scale_min = float(np.nanmin(pd.to_numeric(top["p10"], errors="coerce")))
            scale_max = float(np.nanmax(pd.to_numeric(top["p90"], errors="coerce")))
            if not np.isfinite(scale_min) or not np.isfinite(scale_max) or scale_max <= scale_min:
                scale_min, scale_max = 0.0, 30.0
        else:
            scale_min, scale_max = 0.0, 30.0

        for _, row in top.iterrows():
            SuggestionRow(self.sugg_frame, row, scale_min, scale_max).pack(fill="x", pady=4)

        lineup_txt = "Lineup: " + " ".join(self.lineup_slots) + f"\nPick #{self.current_pick_no} | Rank by: {self.rank_metric.get()}"
        self.lbl_roster.config(text=lineup_txt)

        if self.on_clock:
            self.status.config(text="🟢 On the clock! (Make your pick in Sleeper)")
            beep()
        else:
            self.status.config(text="Listening…")


    def refresh_ui_once(self):
        # clear the suggestion list UI
        for child in self.sugg_frame.winfo_children():
            child.destroy()

        # compute current pick number if you aren’t already
        try:
            self.current_pick_no = len(self.picks) + 1
        except Exception:
            self.current_pick_no = getattr(self, "current_pick_no", 1)

        # rank using roster-aware score
        top_df = self.rank_candidates(topk=getattr(self.args, "topk", 12))
        if top_df is None or top_df.empty:
            return

        # scale for risk bar (handles missing p10/p90 gracefully)
        p10s = pd.to_numeric(top_df.get("p10"), errors="coerce")
        p90s = pd.to_numeric(top_df.get("p90"), errors="coerce")
        if p10s.notna().any() and p90s.notna().any():
            scale_min = float(np.nanmin(p10s))
            scale_max = float(np.nanmax(p90s))
            if scale_max <= scale_min:
                scale_min, scale_max = 0.0, 30.0
        else:
            scale_min, scale_max = 0.0, 30.0

        # paint rows
        for _, row in top_df.iterrows():
            SuggestionRow(self.sugg_frame, row, scale_min, scale_max).pack(fill="x", pady=4)

        # also refresh the right side panels (counts + roster table)
        self.refresh_roster_panel()

    # ---------- actions ----------
    def pick_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if p:
            self.ent_csv.delete(0, "end"); self.ent_csv.insert(0, p)
            self.load_pool(p)
            if self.merged is not None:
                # re-merge projections if already loaded
                self.load_projections(self.ent_ranks.get().strip())

    def pick_ranks(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if p:
            self.ent_ranks.delete(0, "end"); self.ent_ranks.insert(0, p)
            self.load_projections(p)

    def rebuild_projections(self):
        # Run build_projections.py with GUI knobs, then reload projections
        league_id = None
        try:
            draft = get_json(f"{BASE}/draft/{self.ent_draft.get().strip()}")
            league_id = draft.get("league_id")
        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch draft/league: {e}")
            return
        if not league_id:
            messagebox.showerror("Error", "Draft missing league_id. Open your draft room and try again.")
            return

        pool_csv = self.ent_csv.get().strip()
        players_json_guess = "players_nfl.json"
        # let user pick if not found
        if not os.path.exists(players_json_guess):
            pj = filedialog.askopenfilename(title="Pick players_nfl.json", filetypes=[("JSON","*.json")])
            if not pj: return
            players_json_guess = pj

        out_csv = os.path.join(os.path.dirname(pool_csv), "my_projections.csv")
        prior = self.var_prior.get()
        weeks = self.var_weeks.get()
        sims = self.var_sims.get()
        nudge = self.var_nudge.get() / 100.0

        self.status.config(text="⏳ Rebuilding projections…")
        self.btn_connect.config(state="disabled")

        def _run():
            cmd = [
                sys.executable, "build_projections.py",
                "--league-id", str(league_id),
                "--players-json", players_json_guess,
                "--adp-csv", pool_csv,
                "--weeks", str(weeks),
                "--sims", str(sims),
                "--prior-games", str(prior),
                "--adp-nudge", str(nudge),
                "--out", out_csv,
            ]
            try:
                subprocess.run(cmd, check=True)
                self.root.after(0, lambda: self._after_rebuild(out_csv))
            except subprocess.CalledProcessError as e:
                self.root.after(0, lambda: messagebox.showerror("Build failed", str(e)))
            finally:
                self.root.after(0, lambda: (self.status.config(text="Ready"), self.btn_connect.config(state="normal")))

        threading.Thread(target=_run, daemon=True).start()

    def _after_rebuild(self, out_csv):
        self.ent_ranks.delete(0, "end"); self.ent_ranks.insert(0, out_csv)
        self.load_projections(out_csv)
        messagebox.showinfo("Done", f"Projections rebuilt:\n{out_csv}")

    # ----------
    def run(self):
        self.root.mainloop()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-id", required=False, default="")
    ap.add_argument("--username", default=None)
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--csv", default="data/draft_simulation_pool.csv")
    ap.add_argument("--ranks", default=None, help="CSV with season_mean/p10/p90/vorp (from build_projections.py)")
    ap.add_argument("--rank-col", default="vorp", help="'vorp' or 'season_mean'")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--poll", type=float, default=1.5)
    # default knobs (used only when rebuilding)
    ap.add_argument("--prior-games", type=int, default=6)
    ap.add_argument("--weeks", type=int, default=14)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--adp-nudge", type=float, default=10.0)  # %
    ap.add_argument("--debug", action="store_true", help="Print ranking/filters debug info")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = App(args)
    app.run()
