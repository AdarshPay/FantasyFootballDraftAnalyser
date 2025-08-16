# build_projections.py
import argparse
import json
import math
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

BASE = "https://api.sleeper.app/v1"

POS_STARTERS_DEFAULT = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1}

import math

POS_STARTERS_DEFAULT = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1}

def start_prob_from_adp(pos: str, pos_rank: int, teams: int) -> float:
    # Reasonable piecewise priors by position
    if pos == "QB":
        if pos_rank <= teams: return 0.95
        if pos_rank <= teams*1.5: return 0.60
        if pos_rank <= teams*2: return 0.35
        return 0.15
    elif pos in ("TE", "K", "DEF"):
        if pos_rank <= teams: return 0.90
        if pos_rank <= teams*1.5: return 0.60
        return 0.30
    else:  # RB/WR
        # RB ~2 starters/team; WR ~3
        spt = 2 if pos == "RB" else 3
        if pos_rank <= teams*spt: return 0.90
        if pos_rank <= teams*(spt+1): return 0.70
        if pos_rank <= teams*(spt+2): return 0.50
        return 0.30

def depth_chart_factor(order: int | None, pos: str) -> float:
    # Conservative default if unknown
    if order is None: return 1.0
    try: order = int(order)
    except: return 1.0
    if pos == "QB":
        return 1.0 if order == 1 else 0.20
    if pos == "RB":
        if order <= 2: return 1.0
        if order == 3: return 0.70
        return 0.50
    if pos == "WR":
        if order <= 3: return 1.0
        if order == 4: return 0.70
        return 0.50
    return 1.0

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def get_json(url: str, retries: int = 5, pause: float = 0.35):
    for i in range(retries):
        r = requests.get(url, timeout=20)
        if r.ok:
            return r.json()
        time.sleep(pause * (1.5 ** i))
    r.raise_for_status()


def get_league_chain(latest_league_id: str, max_back_seasons: int = 2) -> List[str]:
    chain = []
    cur = latest_league_id
    for _ in range(max_back_seasons):
        lg = get_json(f"{BASE}/league/{cur}")
        chain.append(cur)
        prev = lg.get("previous_league_id")
        if not prev:
            break
        cur = str(prev)
    return chain  # [this_season, previous_season, ...]


def league_lineup_and_size(league_id: str) -> Tuple[List[str], int]:
    lg = get_json(f"{BASE}/league/{league_id}")
    lineup = lg.get("roster_positions", [])
    teams = int(lg.get("total_rosters", 10))
    return lineup, teams


def weeks_for_season(league_id: str) -> List[int]:
    weeks = []
    for wk in range(1, 19):
        url = f"{BASE}/league/{league_id}/matchups/{wk}"
        data = get_json(url)
        if isinstance(data, list) and len(data) > 0:
            weeks.append(wk)
    return weeks


def collect_weekly_points(league_id: str) -> pd.DataFrame:
    lg = get_json(f"{BASE}/league/{league_id}")
    season = int(lg.get("season")) if lg.get("season") else None
    weekly_rows = []
    weeks = weeks_for_season(league_id)

    for wk in weeks:
        data = get_json(f"{BASE}/league/{league_id}/matchups/{wk}")
        for team_obj in data:
            pp = team_obj.get("players_points")
            if isinstance(pp, dict) and pp:
                for pid, pts in pp.items():
                    if pts is None:
                        continue
                    weekly_rows.append((season, wk, str(pid), float(pts)))
            else:
                starters = team_obj.get("starters") or []
                sp = team_obj.get("starters_points") or []
                for pid, pts in zip(starters, sp):
                    if pid is None or pts is None:
                        continue
                    weekly_rows.append((season, wk, str(pid), float(pts)))

    df = pd.DataFrame(weekly_rows, columns=["season", "week", "player_id", "points"])
    return df


def load_players(players_json_path: str) -> pd.DataFrame:
    with open(players_json_path, "r", encoding="utf-8") as f:
        players = json.load(f)
    rows = []
    for pid, p in players.items():
        if p.get("sport") != "nfl":
            continue
        pos = p.get("position")
        if pos in (None, "IDP"):
            continue
        full = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        team = p.get("team") or p.get("team_abbr")
        rows.append((str(pid), full, pos, team))
    return pd.DataFrame(rows, columns=["player_id", "full_name", "position", "team"])


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"QB", "RB", "WR", "TE", "K", "DEF"}
    return df[df["position"].isin(keep)].copy()


def adp_table(adp_csv: str) -> pd.DataFrame:
    df = pd.read_csv(adp_csv)
    if "full_name" not in df.columns:
        for c in df.columns:
            if c.lower() == "player":
                df = df.rename(columns={c: "full_name"})
    if "adp" not in df.columns and "AVG" in df.columns:
        df = df.rename(columns={"AVG": "adp"})
    if "position" in df.columns:
        df["position"] = df["position"].str.upper().replace({"DST": "DEF", "D/ST": "DEF"})
    return df[["full_name", "position", "adp"]].copy()


def make_name_key(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace(",", "")
        .replace(" jr", "")
        .replace(" sr", "")
        .replace(" iii", "")
        .replace(" ii", "")
        .strip()
    )


def attach_meta(df_points: pd.DataFrame, players_df: pd.DataFrame, adp_df: pd.DataFrame) -> pd.DataFrame:
    meta = players_df.copy()
    meta["name_key"] = meta["full_name"].apply(make_name_key)
    adp = adp_df.copy()
    adp["name_key"] = adp["full_name"].apply(make_name_key)

    meta_small = meta[["player_id", "full_name", "position", "team", "name_key"]].drop_duplicates()
    adp_small = adp[["name_key", "position", "adp"]].drop_duplicates()

    m1 = meta_small.merge(adp_small, on=["name_key", "position"], how="left")

    # name-only fill if still missing
    missing = m1["adp"].isna()
    if missing.any():
        name_only = adp.groupby("name_key", as_index=False)["adp"].min()
        m1.loc[missing, "adp"] = m1.loc[missing, "name_key"].map(dict(zip(name_only["name_key"], name_only["adp"])))

    df = df_points.merge(m1[["player_id", "full_name", "position", "team", "adp"]], on="player_id", how="left")
    return df


def empirical_bayes_projection(weekly: pd.DataFrame, prior_games: int = 6) -> pd.DataFrame:
    g = weekly.groupby(["player_id", "position"], as_index=False).agg(
        gp=("points", "count"), mu=("points", "mean"), sigma=("points", "std")
    )
    g["sigma"] = g["sigma"].fillna(0.0)

    pos_stats = weekly.groupby("position").agg(pos_mu=("points", "mean"), pos_sigma=("points", "std")).reset_index()
    pos_stats = pos_stats.set_index("position")

    mu_hat, sigma_hat = [], []
    for _, row in g.iterrows():
        pos = row["position"]
        n = int(row["gp"])
        mu = float(row["mu"])
        s = float(row["sigma"])
        pmu = float(pos_stats.loc[pos, "pos_mu"])
        psig = float(pos_stats.loc[pos, "pos_sigma"]) if not math.isnan(pos_stats.loc[pos, "pos_sigma"]) else 0.0

        post_mu = (n * mu + prior_games * pmu) / max(1, (n + prior_games))
        num = max(n - 1, 0) * (s ** 2) + prior_games * (psig ** 2)
        den = max((n - 1) + prior_games, 1)
        post_sd = math.sqrt(num / den) if den > 0 else psig

        mu_hat.append(post_mu)
        sigma_hat.append(post_sd)

    g["mu_week"] = mu_hat
    g["sd_week"] = sigma_hat
    return g[["player_id", "position", "gp", "mu_week", "sd_week"]]


def season_sim(player_tbl: pd.DataFrame, weeks: int, sims: int, rng: np.random.Generator) -> pd.DataFrame:
    # Broadcast-safe draw: (players, sims, weeks)
    mu = pd.to_numeric(player_tbl["mu_week"], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1, 1)
    sd = pd.to_numeric(player_tbl["sd_week"], errors="coerce").fillna(0.0).to_numpy()
    sd = np.maximum(sd, 0.5).reshape(-1, 1, 1)  # floor volatility

    n_players = mu.shape[0]
    z = rng.standard_normal(size=(n_players, sims, weeks))
    draws = np.clip(z * sd + mu, 0, None).sum(axis=2)

    out = player_tbl.copy()
    out["season_mean"] = draws.mean(axis=1)
    out["season_std"] = draws.std(axis=1)
    out["p10"] = np.percentile(draws, 10, axis=1)
    out["p90"] = np.percentile(draws, 90, axis=1)
    return out


def starters_per_team(lineup: List[str]) -> Counter:
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


def compute_vorp(sim_df: pd.DataFrame, lineup: List[str], teams: int) -> pd.DataFrame:
    sp_team = starters_per_team(lineup)
    starters_league = {pos: int(round(sp_team[pos] * teams)) for pos in ("QB", "RB", "WR", "TE", "K", "DEF")}
    vorp_vals = []
    for pos, grp in sim_df.groupby("position"):
        g = grp.sort_values("season_mean", ascending=False).reset_index(drop=True)
        repl_idx = max(starters_league.get(pos, 0) - 1, 0)
        repl_pts = float(g.loc[repl_idx, "season_mean"]) if len(g) else 0.0
        for _, row in g.iterrows():
            vorp_vals.append((row["player_id"], row["season_mean"] - repl_pts))
    vorp_df = pd.DataFrame(vorp_vals, columns=["player_id", "vorp"])
    return sim_df.merge(vorp_df, on="player_id", how="left")


def main():
    ap = argparse.ArgumentParser(description="Build projections & rankings from your Sleeper league history.")
    ap.add_argument("--league-id", required=True)
    ap.add_argument("--players-json", default="players_nfl.json")
    ap.add_argument("--adp-csv", default="data/draft_simulation_pool.csv")
    ap.add_argument("--weeks", type=int, default=14)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prior-games", type=int, default=6, help="Empirical Bayes prior strength (games).")
    ap.add_argument("--adp-nudge", type=float, default=0.10, help="±percent bump to weekly mean by ADP rank within position (0.0–0.3).")
    ap.add_argument("--out", default="data/my_projections.csv")
    ap.add_argument("--teams", type=int, default=10,
    help="League size to calibrate start probabilities (default 10).")
    ap.add_argument("--playing-time-adjust", "--pta", dest="pta", type=float, default=1.0,
        help="Exponent to penalize mean by availability (1.0 default; higher = stronger penalty).")
    ap.add_argument("--strict-backups-cutoff", type=int, default=None,
        help="If set, drop QBs with pos_rank > cutoff unless depth_chart_order==1 (e.g., 20).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) League chain (this + previous)
    league_chain = get_league_chain(args.league_id, max_back_seasons=2)

    # 2) Weekly points
    weekly_frames = []
    for lid in league_chain:
        dfw = collect_weekly_points(lid)
        if not dfw.empty:
            weekly_frames.append(dfw)
    if not weekly_frames:
        raise SystemExit("No weekly matchup data found.")
    weekly = pd.concat(weekly_frames, ignore_index=True)

    # 3) Metadata & ADP
    players_df = normalize_positions(load_players(args.players_json))
    adp_df = adp_table(args.adp_csv)
    weekly = attach_meta(weekly, players_df, adp_df)
    weekly = weekly.dropna(subset=["position"])

    # 4) Empirical Bayes posterior
    posterior = empirical_bayes_projection(weekly, prior_games=args.prior_games)

    # 5) Join meta + ADP
    meta = players_df[["player_id", "full_name", "position", "team"]].drop_duplicates()
    adp_df["name_key"] = adp_df["full_name"].apply(make_name_key)
    meta["name_key"] = meta["full_name"].apply(make_name_key)
    meta_adp = meta.merge(adp_df[["name_key", "position", "adp"]], on=["name_key", "position"], how="left").drop(columns=["name_key"])

    proj = posterior.merge(meta_adp, on=["player_id", "position"], how="left")

    # 6) Mild ADP-based scaling of weekly mean (±adp_nudge)
    if args.adp_nudge and args.adp_nudge > 0:
        # 0 (best ADP) → +nudge ; 1 (worst ADP) → -nudge
        rank_q = (
            proj.dropna(subset=["adp"])
            .groupby("position")["adp"]
            .transform(lambda s: (s.rank(method="first") - 1) / max(len(s) - 1, 1))
        )
        # scale in [1 - nudge, 1 + nudge]
        scale = 1.0 + (0.5 - rank_q.fillna(0.5)) * (2.0 * args.adp_nudge)
        proj.loc[proj["adp"].notna(), "mu_week"] *= scale
    
    # Make sure ADP is numeric
    proj["adp"] = pd.to_numeric(proj.get("adp"), errors="coerce")

    # Positional ADP rank (1 = best) among rows with ADP; fall back to large number when missing
    proj = proj.sort_values(["position", "adp"])
    proj["pos_rank"] = (
        proj.groupby("position")["adp"]
            .rank(method="first")
            .astype("Int64")
    )
    # Missing ADP → give very large rank so they look like non-starters in start_prob
    proj["pos_rank"] = proj["pos_rank"].fillna(999).astype(int)

    # Merge depth_chart_order from Sleeper metadata
    try:
        # players_meta should be the dict loaded from players_nfl.json
        meta_df = pd.DataFrame.from_dict(players_meta, orient="index")

        # Ensure there's a player_id column to join on
        if "player_id" not in meta_df.columns:
            meta_df["player_id"] = meta_df.index

        keep = ["player_id", "depth_chart_order", "position"]
        keep = [c for c in keep if c in meta_df.columns]
        meta_df = meta_df[keep].copy()

        # Some Sleeper positions can be like "SWR"; we only need depth order numeric
        proj = proj.merge(meta_df[["player_id", "depth_chart_order"]], on="player_id", how="left")
    except Exception:
        # If anything goes wrong, just proceed without depth info
        proj["depth_chart_order"] = np.nan
        
    def _as_series(df: pd.DataFrame, col: str, default_val=np.nan, dtype="float64"):
        """Return a numeric Series for df[col], or a Series full of default_val (aligned to df.index)."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        # fallback series aligned with df index
        return pd.Series([default_val] * len(df), index=df.index, dtype=dtype)

    teams = getattr(args, "teams", 10)          # league size used by start-prob model
    pta   = getattr(args, "pta", 1.0)           # playing-time-adjust exponent
    prior_g = max(1, int(getattr(args, "prior_games", 8)))

    # 1) Games sample factor
    games_recent_ser = _as_series(proj, "games_recent", default_val=np.nan)
    gp_ratio = np.clip(games_recent_ser.fillna(0.0) / float(prior_g), 0.0, 1.0).to_numpy()

    # 2) Start probability from positional ADP rank
    pos_arr  = proj["position"].astype(str).to_numpy()
    pos_rank_ser = _as_series(proj, "pos_rank", default_val=999)
    pos_rank = pos_rank_ser.fillna(999).astype(int).to_numpy()
    sp = np.array([start_prob_from_adp(pos_arr[i], int(pos_rank[i]), teams) for i in range(len(proj))], dtype=float)

    # 3) Depth chart factor
    dc_order_ser = _as_series(proj, "depth_chart_order", default_val=np.nan)
    def _dc(i):
        v = dc_order_ser.iat[i]
        return depth_chart_factor(None if np.isnan(v) else int(v), pos_arr[i])
    dcf = np.array([_dc(i) for i in range(len(proj))], dtype=float)

    # 4) Availability in [0.05, 1.0]
    avail = (0.5 * gp_ratio + 0.5 * sp) * dcf
    avail = np.clip(avail, 0.05, 1.0)

    # Optional: strict cut for deep QB backups unless DC#1
    cutoff = getattr(args, "strict_backups_cutoff", None)
    if cutoff is not None:
        qb = (proj["position"] == "QB")
        deep = pos_rank_ser.fillna(999) > int(cutoff)
        not_starter = dc_order_ser.fillna(1).astype(float) != 1.0
        mask_keep = ~(qb & deep & not_starter)

        proj = proj.loc[mask_keep].reset_index(drop=True)

        # Recompute arrays after filtering
        games_recent_ser = _as_series(proj, "games_recent", default_val=np.nan)
        gp_ratio = np.clip(games_recent_ser.fillna(0.0) / float(prior_g), 0.0, 1.0).to_numpy()

        pos_arr  = proj["position"].astype(str).to_numpy()
        pos_rank_ser = _as_series(proj, "pos_rank", default_val=999)
        pos_rank = pos_rank_ser.fillna(999).astype(int).to_numpy()
        sp = np.array([start_prob_from_adp(pos_arr[i], int(pos_rank[i]), teams) for i in range(len(proj))], dtype=float)

        dc_order_ser = _as_series(proj, "depth_chart_order", default_val=np.nan)
        def _dc2(i):
            v = dc_order_ser.iat[i]
            return depth_chart_factor(None if np.isnan(v) else int(v), pos_arr[i])
        dcf = np.array([_dc2(i) for i in range(len(proj))], dtype=float)

        avail = np.clip((0.5 * gp_ratio + 0.5 * sp) * dcf, 0.05, 1.0)

    # 5) Adjust mu/sd and store
    mu_ser = _as_series(proj, "mu", default_val=0.0)
    sd_ser = _as_series(proj, "sd", default_val=0.0)

    mu_adj = mu_ser.fillna(0.0).to_numpy() * (avail ** pta)
    sd_adj = sd_ser.fillna(0.0).to_numpy() * np.maximum(0.3, avail)

    proj["avail"]  = avail
    proj["mu_adj"] = mu_adj
    proj["sd_adj"] = sd_adj

    # Overwrite μ/σ used by the simulator (simple drop-in)
    proj["mu"] = mu_adj
    proj["sd"] = sd_adj

    # Guard: if filtering removed everyone, skip sim gracefully
    if len(proj) == 0:
        print("⚠️ No players left after filters; writing empty projections.")
        proj.to_csv(args.out, index=False)
        sys.exit(0)

    # >>> Call the simulator as you already do (unchanged) <<<
    sim = season_sim(proj, weeks=args.weeks, sims=args.sims, rng=rng)

        
    # 7) Season simulation
    sim = season_sim(proj, weeks=args.weeks, sims=args.sims, rng=rng)

    # 8) VORP for your lineup
    lineup, teams = league_lineup_and_size(args.league_id)
    sim = compute_vorp(sim.merge(meta_adp, on=["player_id", "position"], how="left"), lineup, teams)

    # 9) Ranks
    sim["pos_rank"] = sim.groupby("position")["season_mean"].rank(ascending=False, method="first")
    sim["overall_rank"] = sim["season_mean"].rank(ascending=False, method="first")

    out = sim.merge(meta_adp, on=["player_id", "position"], how="left")
    out = out[[
        "player_id","full_name","team","position","adp",
        "gp","mu_week","sd_week",
        "season_mean","season_std","p10","p90","vorp","pos_rank","overall_rank"
    ]].sort_values("overall_rank")
    out.to_csv(args.out, index=False)
    print(f"✅ Wrote projections → {args.out} (players: {len(out)})")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
