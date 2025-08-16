# live_assistant_gui.py
# Sleeper Draft Assistant with a Tkinter GUI (works for live drafts and mocks)
# Usage:
#   python live_assistant_gui.py --draft-id <DRAFT_ID> --username <YourSleeperUsername> --csv data/draft_simulation_pool.csv
#   python live_assistant_gui.py --draft-id <DRAFT_ID> --slot 1

import argparse
import threading
import time
import queue
import platform
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import requests
import tkinter as tk
from tkinter import ttk, messagebox

API = "https://api.sleeper.app/v1"
UA  = "hybrid-live-assistant-gui/1.0"

CORE_POS = ["QB","RB","WR","TE","K","DEF"]
FLEX_MAP = {
    "FLEX":       {"RB","WR","TE"},
    "WRRB":       {"WR","RB"},
    "WRRBTE":     {"WR","RB","TE"},
    "W/R/T":      {"WR","RB","TE"},
    "SUPER_FLEX": {"QB","RB","WR","TE"},
    "SFLX":       {"QB","RB","WR","TE"},
}

# ---------------- HTTP helpers ----------------
def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s

def _get(s: requests.Session, url: str, params=None, sleep=0.12):
    r = s.get(url, params=params, timeout=25)
    if sleep: time.sleep(sleep)
    r.raise_for_status()
    return r.json()

def get_user_id(s: requests.Session, username: str) -> str:
    return str(_get(s, f"{API}/user/{username}")["user_id"])

def get_draft(s: requests.Session, draft_id: str) -> dict:
    return _get(s, f"{API}/draft/{draft_id}")

def get_picks(s: requests.Session, draft_id: str) -> List[dict]:
    return _get(s, f"{API}/draft/{draft_id}/picks")

def get_league(s: requests.Session, league_id: str) -> dict:
    return _get(s, f"{API}/league/{league_id}")

# ---------------- Utils & scoring ----------------
def normalize_pos(pos: str) -> str:
    pos = (pos or "").upper()
    return "DEF" if pos in ("DST","DEF/ST","D/ST") else pos

def compute_next_slot(pick_no: int, teams: int) -> int:
    rnd = (pick_no - 1) // teams + 1
    idx = (pick_no - 1) % teams
    return (idx + 1) if (rnd % 2 == 1) else (teams - idx)

def round_from_pick(pick_no: int, teams: int) -> int:
    return (pick_no - 1)//teams + 1

def adp_window(current_pick: int) -> int:
    if current_pick <= 20:  return 3
    if current_pick <= 40:  return 6
    if current_pick <= 80:  return 10
    return 20

def jitter() -> float:
    return np.random.normal(0, 0.01)

def derive_min_reqs(roster_positions: List[str]) -> Tuple[Dict[str,int], List[Tuple[Set[str],int]]]:
    base = {p:0 for p in CORE_POS}
    flex_slots: List[Tuple[Set[str],int]] = []
    flex_counts: Dict[str,int] = {}
    for rp in roster_positions or []:
        rp = rp.upper()
        if rp in base:
            base[rp] += 1
        elif rp in FLEX_MAP:
            flex_counts[rp] = flex_counts.get(rp, 0) + 1
        else:
            # ignore bench & IDP in mins
            if rp in ("BN","BE","DL","LB","DB","IDP"):
                continue
    for key, cnt in flex_counts.items():
        flex_slots.append((FLEX_MAP[key], cnt))
    return base, flex_slots

def count_positions(team_df: pd.DataFrame) -> Dict[str,int]:
    out = {p:0 for p in CORE_POS}
    for p in team_df["position"]:
        if p in out: out[p] += 1
    return out

def soft_caps_from_mins(mins: Dict[str,int]) -> Dict[str,int]:
    caps = {}
    caps["QB"]  = mins.get("QB",0)  + 1
    caps["TE"]  = mins.get("TE",0)  + 1
    caps["RB"]  = mins.get("RB",0)  + 3
    caps["WR"]  = mins.get("WR",0)  + 3
    caps["K"]   = mins.get("K",0)   + 0
    caps["DEF"] = mins.get("DEF",0) + 0
    return caps

def can_satisfy_with_flex(mins: Dict[str,int], flexes: List[Tuple[Set[str],int]], counts: Dict[str,int]) -> Dict[str,int]:
    deficit = {p: max(0, mins.get(p,0) - counts.get(p,0)) for p in CORE_POS}
    for elig, cnt in flexes:
        for _ in range(cnt):
            pick_pos, max_def = None, 0
            for p in elig:
                if p in deficit and deficit[p] > max_def:
                    pick_pos, max_def = p, deficit[p]
            if pick_pos:
                deficit[pick_pos] -= 1
    eff_mins = {p: counts.get(p,0) + max(0, deficit.get(p,0)) for p in CORE_POS}
    return eff_mins

def allowed_positions(counts: Dict[str,int], eff_min_targets: Dict[str,int], soft_caps: Dict[str,int], round_no: int) -> Set[str]:
    must_fill: Set[str] = set()
    for p in CORE_POS:
        if counts.get(p,0) < eff_min_targets.get(p,0):
            must_fill.add(p)
    if must_fill:
        if round_no < 12:
            non_sp = {p for p in must_fill if p not in {"K","DEF"}}
            if non_sp:
                return non_sp
        return must_fill
    ok = set(CORE_POS)
    for p, cap in soft_caps.items():
        if counts.get(p,0) >= cap:
            ok.discard(p)
    if round_no < 12:
        ok -= {"K","DEF"}
    if not ok:
        ok = {"RB","WR"}
    return ok

def scarcity_features(avail_df: pd.DataFrame) -> Dict[str,int]:
    top = avail_df.nsmallest(50, "adp")
    return {f"remain_{p}_top50": int((top["position"] == p).sum()) for p in CORE_POS}

def score_candidate(row, *, current_pick, round_number, team_counts, scarcity, league_size):
    adp = float(row["adp"])
    pos = row["position"]
    window = max(1.0, adp_window(current_pick))
    adp_component = -abs(adp - current_pick) / window
    bonus = 0.0
    if pos in ("RB","WR") and round_number <= 10:
        bonus += 0.25
    scarcity_key = f"remain_{pos}_top50"
    sc = scarcity.get(scarcity_key, 0)
    if pos in ("RB","WR","TE") and sc <= 6:
        bonus += (6 - sc) * 0.10
    two_round = 2 * league_size
    until_next = two_round - ((current_pick - 1) % two_round)
    if until_next >= league_size and pos in ("RB","WR","TE"):
        bonus += 0.2
    return adp_component + bonus + jitter()

def build_suggestions(pool: pd.DataFrame,
                      drafted_ids: Set[str],
                      my_ids: Set[str],
                      league_size: int,
                      rounds: int,
                      roster_positions: List[str],
                      next_pick_no: int,
                      topk: int = 8) -> Tuple[pd.DataFrame, Dict[str,int], List[str]]:
    round_no = round_from_pick(next_pick_no, league_size)
    my_df = pool[pool["player_id"].isin(my_ids)]
    counts = count_positions(my_df)
    mins, flexes = derive_min_reqs(roster_positions)
    soft_caps = soft_caps_from_mins(mins)
    eff_targets = can_satisfy_with_flex(mins, flexes, counts)

    # Need K/DEF late if league actually has them
    need_msgs = []
    need_specialists = []
    if round_no >= 13:
        for sp in ("K","DEF"):
            if mins.get(sp,0) > 0 and counts.get(sp,0) < mins.get(sp,0):
                need_specialists.append(sp)
    avail_all = pool[~pool["player_id"].isin(drafted_ids)].copy()

    if need_specialists:
        df = avail_all[avail_all["position"].isin(need_specialists)].sort_values("adp").head(topk).copy()
        if not df.empty:
            df["reason"] = [f"Lineup requires {p}; best remaining {p} by ADP" for p in df["position"]]
            need_msgs = [f"Need {', '.join(need_specialists)} to satisfy lineup."]
            return df[["full_name","position","team","adp","player_id","reason"]], counts, need_msgs

    allow = allowed_positions(counts, eff_targets, soft_caps, round_no)
    avail = avail_all[avail_all["position"].isin(allow)].copy()
    window = adp_window(next_pick_no)
    within = avail[avail["adp"] <= next_pick_no + window].copy()
    candidates = within if not within.empty else avail.nsmallest(40, "adp").copy()

    scarcity = scarcity_features(avail_all)
    scores = []
    for _, row in candidates.iterrows():
        s = score_candidate(row,
                            current_pick=next_pick_no,
                            round_number=round_no,
                            team_counts=counts,
                            scarcity=scarcity,
                            league_size=league_size)
        scores.append(s)
    out = candidates.assign(score=scores).sort_values("score", ascending=False).head(topk)

    reasons = []
    for _, r in out.iterrows():
        bits = [f"ADP {r['adp']:.1f} near pick {next_pick_no}"]
        if r["position"] in ("RB","WR","TE"):
            sc = scarcity.get(f"remain_{r['position']}_top50", None)
            if sc is not None and sc <= 6:
                bits.append(f"scarce {r['position']} (top50 remain {sc})")
        reasons.append("; ".join(bits))
    out["reason"] = reasons
    return out[["full_name","position","team","adp","score","player_id","reason"]], counts, need_msgs

# ---------------- Worker thread ----------------
class DraftWorker(threading.Thread):
    def __init__(self, args, pool_df: pd.DataFrame, outq: queue.Queue):
        super().__init__(daemon=True)
        self.args = args
        self.pool = pool_df
        self.outq = outq
        self.stop_flag = threading.Event()
        self.s = _session()
        self.my_user_id = None
        self.my_slot = None
        self.my_roster_id = None
        self.teams = None
        self.rounds = None
        self.roster_positions = []
        self.drafted_ids: Set[str] = set()
        self.my_ids: Set[str] = set()
        self.seen_pick_nos: Set[int] = set()
        self.last_prompted_pick = None

    def resolve_ids(self):
        draft = get_draft(self.s, self.args.draft_id)
        self.teams  = int(draft["settings"]["teams"])
        self.rounds = int(draft["settings"]["rounds"])
        league_id = draft.get("league_id")

        if league_id:
            try:
                league = get_league(self.s, str(league_id))
                self.roster_positions = [str(x).upper() for x in league.get("roster_positions", [])]
            except Exception:
                self.roster_positions = [str(x).upper() for x in draft.get("metadata", {}).get("roster_positions", [])]

        # my slot
        self.my_user_id = None
        self.my_slot = None
        if self.args.username:
            try:
                self.my_user_id = get_user_id(self.s, self.args.username)
                self.my_slot = int(draft.get("draft_order", {}).get(self.my_user_id))
            except Exception:
                self.my_user_id = None
                self.my_slot = None
        if not self.my_slot and self.args.slot:
            self.my_slot = int(self.args.slot)
        if not self.my_slot:
            raise RuntimeError("Couldn't resolve your slot. Pass --username or --slot.")

        slot_to_roster = {int(k): int(v) for k, v in draft.get("slot_to_roster_id", {}).items() if v is not None}
        self.my_roster_id = slot_to_roster.get(self.my_slot, self.my_slot)

        # announce
        msg = {
            "type": "status",
            "text": f"Connected: teams={self.teams}, rounds={self.rounds}, your_slot={self.my_slot}, roster_id={self.my_roster_id}",
            "lineup": ", ".join(self.roster_positions) if self.roster_positions else "(no lineup in API)"
        }
        self.outq.put(msg)

    def run(self):
        try:
            self.resolve_ids()
        except Exception as e:
            self.outq.put({"type":"error","text":f"Startup error: {e}"})
            return

        poll = float(self.args.poll)

        while not self.stop_flag.is_set():
            try:
                picks = get_picks(self.s, self.args.draft_id)
                picks = sorted([p for p in picks if p.get("pick_no") is not None], key=lambda x: x["pick_no"])

                # Update drafted & my roster
                new_recent = []
                for p in picks:
                    pk = p.get("pick_no")
                    if pk in self.seen_pick_nos:
                        continue
                    pid = p.get("player_id")
                    if pid:
                        self.drafted_ids.add(str(pid))
                        # recent pick line
                        name = self.pool.loc[self.pool["player_id"] == str(pid)]["full_name"]
                        name = name.iloc[0] if not name.empty else f"player_id {pid}"
                        new_recent.append(f"{pk}: {name}")

                        roster_id = p.get("roster_id")
                        draft_slot = p.get("draft_slot")
                        picked_by  = p.get("picked_by")

                        is_ours = False
                        if roster_id is not None and int(roster_id) == int(self.my_roster_id):
                            is_ours = True
                        elif draft_slot is not None and int(draft_slot) == int(self.my_slot):
                            is_ours = True
                        elif self.my_user_id and picked_by and str(picked_by) == str(self.my_user_id):
                            is_ours = True

                        if is_ours:
                            self.my_ids.add(str(pid))
                    self.seen_pick_nos.add(pk)

                next_pick_no = (picks[-1]["pick_no"] + 1) if picks else 1
                if next_pick_no > self.teams * self.rounds:
                    self.outq.put({"type":"status","text":"Draft complete. Good luck!"})
                    break

                rnd = round_from_pick(next_pick_no, self.teams)
                nxt_slot = compute_next_slot(next_pick_no, self.teams)
                on_clock = (nxt_slot == self.my_slot)

                # Build suggestions always (so UI shows live board), but we only beep when on the clock & new pick
                recs, counts, needs = build_suggestions(
                    self.pool, self.drafted_ids, self.my_ids, self.teams, self.rounds,
                    self.roster_positions, next_pick_no, topk=int(self.args.topk)
                )

                payload = {
                    "type": "tick",
                    "on_clock": on_clock,
                    "round": rnd,
                    "pick_no": next_pick_no,
                    "slot_on": nxt_slot,
                    "recs": recs,
                    "counts": counts,
                    "recent": new_recent[-10:],  # last few this poll
                    "needs": needs,
                }
                # avoid spamming "on the clock" for same pick
                if on_clock and self.last_prompted_pick != next_pick_no:
                    payload["beep"] = True
                    self.last_prompted_pick = next_pick_no
                self.outq.put(payload)

            except Exception as e:
                self.outq.put({"type":"error","text":str(e)})

            time.sleep(poll)

    def stop(self):
        self.stop_flag.set()

# ---------------- GUI ----------------
class App(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("Sleeper Draft Assistant")
        self.geometry("980x700")
        self.minsize(900, 640)
        self.args = args
        self.outq = queue.Queue()
        self.worker = None
        self.sound_enabled = tk.BooleanVar(value=True)

        self._build_ui()
        self._start_worker()
        self.after(300, self._drain_queue)

    def _build_ui(self):
        # Header
        top = ttk.Frame(self, padding=(10,10))
        top.pack(fill="x")

        self.status_lbl = ttk.Label(top, text="Connecting...", font=("Segoe UI", 11, "bold"))
        self.status_lbl.pack(side="left")

        self.lineup_lbl = ttk.Label(top, text="", foreground="#666")
        self.lineup_lbl.pack(side="left", padx=(12,0))

        ttk.Checkbutton(top, text="Sound", variable=self.sound_enabled).pack(side="right")
        ttk.Label(top, text=f"poll {self.args.poll}s | top {self.args.topk}", foreground="#666").pack(side="right", padx=10)

        # Middle: left suggestions, right roster + recent
        mid = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        # Suggestions pane
        left = ttk.Labelframe(mid, text="Suggestions", padding=6)
        self.tree = ttk.Treeview(left, columns=("pos","adp","reason"), show="headings", height=18)
        self.tree.heading("pos", text="Pos")
        self.tree.heading("adp", text="ADP")
        self.tree.heading("reason", text="Why")
        self.tree.column("pos", width=60, anchor="center")
        self.tree.column("adp", width=70, anchor="center")
        self.tree.column("reason", width=550, anchor="w")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        mid.add(left, weight=3)

        # Right pane: roster + recent
        right = ttk.Labelframe(mid, text="Your Roster & Feed", padding=6)
        # Roster grid
        self.roster_grid = ttk.Treeview(right, columns=("have","need"), show="headings", height=6)
        self.roster_grid.heading("have", text="Have")
        self.roster_grid.heading("need", text="Min")
        self.roster_grid.column("have", width=70, anchor="center")
        self.roster_grid.column("need", width=70, anchor="center")
        ttk.Label(right, text="Positions", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.roster_grid.grid(row=1, column=0, sticky="ew", pady=(0,10))

        # Needs
        self.needs_lbl = ttk.Label(right, text="", foreground="#c00")
        self.needs_lbl.grid(row=2, column=0, sticky="w", pady=(0,10))

        # On-clock banner
        self.banner = ttk.Label(right, text="", font=("Segoe UI", 10, "bold"))
        self.banner.grid(row=3, column=0, sticky="w", pady=(0,10))

        # Recent picks
        ttk.Label(right, text="Recent picks (latest poll):", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w")
        self.recent_box = tk.Text(right, height=10, width=30, wrap="word")
        self.recent_box.grid(row=5, column=0, sticky="nsew")
        right.rowconfigure(5, weight=1)
        right.columnconfigure(0, weight=1)

        mid.add(right, weight=2)

        # Footer
        bottom = ttk.Frame(self, padding=(10,0,10,10))
        bottom.pack(fill="x")
        self.err_lbl = ttk.Label(bottom, text="", foreground="#a00")
        self.err_lbl.pack(side="left")

    def _start_worker(self):
        # Load pool CSV here so GUI can report fast errors
        try:
            pool = pd.read_csv(self.args.csv)
        except Exception as e:
            messagebox.showerror("CSV error", f"Failed to read {self.args.csv}\n{e}")
            self.destroy()
            return
        req = {"player_id","full_name","position","adp"}
        if not req.issubset(pool.columns):
            messagebox.showerror("CSV error", f"CSV must have columns: {', '.join(sorted(req))}")
            self.destroy()
            return
        pool["player_id"] = pool["player_id"].astype(str)
        pool["position"]  = pool["position"].map(normalize_pos)
        pool["adp"]       = pool["adp"].astype(float)

        self.worker = DraftWorker(self.args, pool, self.outq)
        self.worker.start()

    def _drain_queue(self):
        try:
            while True:
                msg = self.outq.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.after(300, self._drain_queue)

    def _handle_msg(self, msg):
        t = msg.get("type")
        if t == "status":
            self.status_lbl.config(text=msg.get("text", ""))
            lineup = msg.get("lineup", "")
            if lineup:
                self.lineup_lbl.config(text=f" | Lineup: {lineup}")
        elif t == "error":
            self.err_lbl.config(text=msg.get("text","(unknown error)"))
        elif t == "tick":
            self.err_lbl.config(text="")
            rnd = msg.get("round")
            pk  = msg.get("pick_no")
            slot_on = msg.get("slot_on")
            on_clock = msg.get("on_clock")

            self.banner.config(text=f"Round {rnd} | Upcoming pick #{pk} | Slot on clock: {slot_on}")
            if on_clock:
                self.banner.config(foreground="#0a0")
                if msg.get("beep") and self.sound_enabled.get():
                    self._beep()
            else:
                self.banner.config(foreground="#333")

            # Update suggestions table
            for i in self.tree.get_children():
                self.tree.delete(i)
            recs = msg.get("recs")
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                for _, r in recs.iterrows():
                    name = str(r["full_name"])
                    pos  = str(r["position"])
                    adp  = f"{float(r['adp']):.1f}"
                    why  = str(r.get("reason",""))
                    self.tree.insert("", "end", values=(pos, adp, f"{name} — {why}"))

            # Update roster
            counts = msg.get("counts", {})
            # Derive mins so we can show required
            # We don't have mins directly, but worker knows roster_positions; recompute
            # Not passed in tick to keep msg small; instead show 'have' only:
            for i in self.roster_grid.get_children():
                self.roster_grid.delete(i)
            # show have and blank need; the needs text will show K/DEF reminders
            for p in CORE_POS:
                have = counts.get(p,0)
                self.roster_grid.insert("", "end", values=(have, ""))

            needs = msg.get("needs", [])
            self.needs_lbl.config(text=("; ".join(needs)) if needs else "")

            # Update recent picks
            recent = msg.get("recent", [])
            if recent:
                self.recent_box.delete("1.0", tk.END)
                self.recent_box.insert(tk.END, "\n".join(recent))
        else:
            pass

    def _beep(self):
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
            else:
                print("\a", end="", flush=True)  # terminal bell
        except Exception:
            pass

    def on_close(self):
        try:
            if self.worker:
                self.worker.stop()
        finally:
            self.destroy()

def parse_args():
    ap = argparse.ArgumentParser(description="Sleeper Draft Assistant GUI")
    ap.add_argument("--draft-id", required=True, help="Sleeper draft_id (real or mock)")
    ap.add_argument("--username", help="Your Sleeper username (recommended)")
    ap.add_argument("--slot", type=int, help="Your draft slot if not using --username")
    ap.add_argument("--csv", default="data/draft_simulation_pool.csv", help="Merged player pool CSV")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--poll", type=float, default=1.5, help="Poll interval seconds")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = App(args)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
