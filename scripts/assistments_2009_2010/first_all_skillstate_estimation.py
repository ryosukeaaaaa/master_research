"""
Estimate current/future skill states using DINA on ASSISTments Skill Builder 2009-2010.

Design (agreed):
- Item = template_id
- Q-row for each template_id is the most frequent skill_id-set across problem_id within that template
- Target skills are provided as skill_id list (--skills ...)
- Filter logs to target skills only, build per-user timeline by order_id
- Split per-user timeline into first half / full
- Valid users: first half contains all target skills
- For each period, compress multiple attempts of same (user, template) by taking the last (max order_id)
- Train DINA once on FULL data to estimate slip/guess
- Fix slip/guess and infer per-student MAP mastery vectors (alpha_current from first half, alpha_future from full)
- Output alpha_current/alpha_future as {0,1}^K (MAP)

Usage (minimal):
  python scripts/estimate_current_future_dina.py --skills 70 77 280 123 456

Defaults:
- data_path: data/raw/assistments_2009_2010/skill_builder_data.csv
- output_dir: data/processed/assistments_2009_2010/first_all_dina_estimation/s{...}/
- epoch: 30
- epsilon: 1e-3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

# EduCDM (as you already use)
from EduCDM import EMDINA


# -----------------------------
# I/O and preprocessing
# -----------------------------
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="latin1")
    needed = ["user_id", "order_id", "problem_id", "template_id", "skill_id", "correct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[needed].dropna()

    # Normalize types
    df["user_id"] = df["user_id"].astype(int)
    df["order_id"] = df["order_id"].astype(int)
    df["problem_id"] = df["problem_id"].astype(int)
    df["template_id"] = df["template_id"].astype(int)
    df["skill_id"] = df["skill_id"].astype(int)
    df["correct"] = df["correct"].astype(int)

    return df


def compute_most_common_skillset_per_template(
    df: pd.DataFrame,
) -> Dict[int, FrozenSet[int]]:
    """
    For each template_id:
      - Build mapping: problem_id -> frozenset(skill_id)
      - Count frequency of each frozenset across problem_id
      - Pick the most frequent as representative (most_common_skills)

    NOTE: df should already be filtered to target skills if you want Q only within target skills.
    """
    template_to_skillset: Dict[int, FrozenSet[int]] = {}

    for template_id, tdf in df.groupby("template_id"):
        # problem_id -> frozenset(skills)
        p2skills = (
            tdf.groupby("problem_id")["skill_id"]
            .apply(lambda x: frozenset(x.values.tolist()))
            .to_dict()
        )

        # Count skill-set frequency across problems
        counts: Dict[FrozenSet[int], int] = {}
        for skills in p2skills.values():
            counts[skills] = counts.get(skills, 0) + 1

        most_common_skills, _ = max(counts.items(), key=lambda kv: kv[1])
        template_to_skillset[int(template_id)] = most_common_skills

    return template_to_skillset


def split_user_half(df_user: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a user's rows into first half and second half by order_id."""
    df_user = df_user.sort_values("order_id")
    n = len(df_user)
    mid = n // 2
    return df_user.iloc[:mid].copy(), df_user.iloc[mid:].copy()


def select_valid_users_first_half_covers_all_skills(
    df_skill_only: pd.DataFrame,
    target_skills: List[int],
) -> List[int]:
    """
    Valid users are those whose FIRST HALF (within skill-only timeline) contains all target skills.
    """
    skill_set = set(target_skills)
    valid = []
    for uid, udf in df_skill_only.groupby("user_id"):
        first, _ = split_user_half(udf)
        if skill_set.issubset(set(first["skill_id"].unique().tolist())):
            valid.append(int(uid))
    return sorted(valid)


def compress_last_response(df_period: pd.DataFrame) -> pd.DataFrame:
    """
    Within a given period dataframe, keep last attempt per (user_id, template_id),
    defined by max(order_id).
    """
    # Ensure stable: sort then take tail(1)
    idx = (
        df_period.sort_values("order_id")
        .groupby(["user_id", "template_id"])
        .tail(1)
        .index
    )
    return df_period.loc[idx].copy()


def build_response_matrix(
    df_period_last: pd.DataFrame,
    users: List[int],
    templates: List[int],
) -> np.ndarray:
    """
    Build response matrix R with shape (n_users, n_templates).
    Values in {0,1}, missing = -1.

    df_period_last must be unique per (user_id, template_id).
    """
    u2i = {u: i for i, u in enumerate(users)}
    t2j = {t: j for j, t in enumerate(templates)}

    R = np.full((len(users), len(templates)), -1, dtype=int)

    for row in df_period_last.itertuples(index=False):
        uid = int(row.user_id)
        tid = int(row.template_id)
        if uid in u2i and tid in t2j:
            R[u2i[uid], t2j[tid]] = int(row.correct)

    return R


# -----------------------------
# Fixed-parameter MAP inference (E-step equivalent)
# -----------------------------
def precompute_state_space(K: int) -> np.ndarray:
    """Return array of integer bitmasks for all 2^K states."""
    n_states = 1 << K
    return np.arange(n_states, dtype=np.uint32)


def templates_to_required_masks(
    templates: List[int],
    template_to_skillset: Dict[int, FrozenSet[int]],
    skill_to_index: Dict[int, int],
) -> np.ndarray:
    """
    Convert each template's required skill-set into a bitmask over K skills.
    """
    req_masks = np.zeros(len(templates), dtype=np.uint32)
    for j, tid in enumerate(templates):
        skills = template_to_skillset[tid]
        mask = 0
        for s in skills:
            # skills are guaranteed to be in skill_to_index (by construction/filtering)
            mask |= (1 << skill_to_index[s])
        req_masks[j] = np.uint32(mask)
    return req_masks


def infer_map_mastery(
    R: np.ndarray,
    req_masks: np.ndarray,
    slip: np.ndarray,
    guess: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Fixed slip/guess, infer per-student MAP mastery vector (0/1)^K.

    R: (n_users, n_templates) values in {0,1,-1}
    req_masks: (n_templates,) bitmask for required skills of each template
    slip, guess: (n_templates,)
    Returns: (n_users, K) binary mastery (MAP state)
    """
    n_users, n_templates = R.shape
    states = precompute_state_space(K)  # (n_states,)
    n_states = len(states)

    # eta: (n_states, n_templates) True iff state satisfies required skill-mask (AND)
    eta = (states[:, None] & req_masks[None, :]) == req_masks[None, :]

    # Precompute log-probs per template for eta=1 / eta=0
    eps = 1e-12
    slip = np.clip(np.asarray(slip, dtype=float), eps, 1 - eps)
    guess = np.clip(np.asarray(guess, dtype=float), eps, 1 - eps)

    log_p_x1_eta1 = np.log(1.0 - slip)   # P(x=1 | eta=1)
    log_p_x0_eta1 = np.log(slip)         # P(x=0 | eta=1)
    log_p_x1_eta0 = np.log(guess)        # P(x=1 | eta=0)
    log_p_x0_eta0 = np.log(1.0 - guess)  # P(x=0 | eta=0)

    mastery_bits = np.zeros((n_users, K), dtype=int)

    for i in range(n_users):
        obs_idx = np.where(R[i] != -1)[0]
        if obs_idx.size == 0:
            mastery_bits[i, :] = 0
            continue

        x = R[i, obs_idx]  # {0,1}

        ll = np.zeros(n_states, dtype=float)

        # Contributions where x=1
        ones_pos = np.where(x == 1)[0]
        if ones_pos.size > 0:
            idx = obs_idx[ones_pos]
            eta_sub = eta[:, idx]
            ll += (eta_sub * log_p_x1_eta1[idx] + (~eta_sub) * log_p_x1_eta0[idx]).sum(axis=1)

        # Contributions where x=0
        zeros_pos = np.where(x == 0)[0]
        if zeros_pos.size > 0:
            idx = obs_idx[zeros_pos]
            eta_sub = eta[:, idx]
            ll += (eta_sub * log_p_x0_eta1[idx] + (~eta_sub) * log_p_x0_eta0[idx]).sum(axis=1)

        best_state = int(np.argmax(ll))
        for k in range(K):
            mastery_bits[i, k] = (best_state >> k) & 1

    return mastery_bits


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estimate current/future skill states using DINA (template_id items) on ASSISTments Skill Builder 2009-2010."
    )

    parser.add_argument(
        "--skills",
        type=int,
        nargs="+",
        required=True,
        help="Target skill_ids (space-separated). Example: --skills 70 77 280",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/assistments_2009_2010/skill_builder_data.csv",
        help="Path to skill_builder_data.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, auto-generated as:\n"
            "  data/processed/assistments_2009_2010/first_all_dina_estimation/s{skill_ids...}/"
        ),
    )
    parser.add_argument("--epoch", type=int, default=30, help="EM training epochs for EMDINA (default: 30)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Convergence epsilon for EMDINA (default: 1e-3)")

    args = parser.parse_args()

    target_skills = sorted(list(set(int(s) for s in args.skills)))
    K = len(target_skills)
    if K == 0:
        raise ValueError("No skills provided.")
    if K > 20:
        print("[WARN] K>20 makes exact MAP over 2^K expensive. Recommended K<=15 (or be prepared for slower inference).")

    # Auto output dir
    base_output_root = Path("data/processed/assistments_2009_2010/first_all_dina_estimation")
    skill_str = "_".join([f"s{s}" for s in target_skills])
    out = Path(args.output_dir) if args.output_dir is not None else (base_output_root / skill_str)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DINA CURRENT/FUTURE ESTIMATION (template_id items)")
    print("=" * 80)
    print(f"Skills (K={K}): {target_skills}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {out.resolve()}")
    print(f"EMDINA epoch={args.epoch}, epsilon={args.epsilon}")
    print("=" * 80)

    print("Loading data...")
    df = load_data(args.data_path)

    print("Filtering to target skills only...")
    df_skill = df[df["skill_id"].isin(target_skills)].copy()
    df_skill = df_skill.sort_values(["user_id", "order_id"]).reset_index(drop=True)

    print(f"  Records (skill-only): {len(df_skill):,}")
    print(f"  Users (skill-only): {df_skill['user_id'].nunique():,}")

    if len(df_skill) == 0:
        raise ValueError("No records after filtering to target skills. Check your skill_ids.")

    print("Computing template_id -> most_common_skills (based on problem_id skill-set frequency)...")
    template_to_skillset = compute_most_common_skillset_per_template(df_skill)

    # Keep templates whose most_common_skills are non-empty and subset of target skills
    target_set = set(target_skills)
    templates_all = sorted([
        tid for tid, ss in template_to_skillset.items()
        if len(ss) > 0 and set(ss).issubset(target_set)
    ])

    print(f"  Templates usable (within target skills): {len(templates_all)}")

    print("Selecting valid users: first half covers all target skills...")
    valid_users = select_valid_users_first_half_covers_all_skills(df_skill, target_skills)
    print(f"  Valid users: {len(valid_users):,}")

    if len(valid_users) == 0:
        raise ValueError(
            "No valid users found under the 'first half covers all target skills' constraint.\n"
            "Try different skills, reduce K, or relax the user selection rule."
        )

    # Save config-ish info
    with open(out / "run_config.json", "w") as f:
        json.dump(
            {
                "data_path": args.data_path,
                "skills": target_skills,
                "K": K,
                "epoch": args.epoch,
                "epsilon": args.epsilon,
                "output_dir": str(out),
                "selection_rule": "valid_users: first half (within skill-only timeline) covers all target skills",
                "item": "template_id",
                "q_rule": "most_common_skillset_per_template (problem_id skillset frequency)",
                "repeat_rule": "last attempt within period (max order_id) per (user, template)",
                "alpha_rule": "MAP over all 2^K states with slip/guess fixed (learned on full)",
            },
            f,
            indent=2,
        )

    # Save valid users
    pd.DataFrame({"user_id": valid_users}).to_csv(out / "valid_users.csv", index=False)

    # Build per-user first half and full data for valid users
    df_valid = df_skill[df_skill["user_id"].isin(valid_users)].copy()

    print("Splitting valid users into first-half (per user, on skill-only timeline)...")
    first_parts = []
    for uid, udf in df_valid.groupby("user_id"):
        first, _ = split_user_half(udf)
        first_parts.append(first)
    df_first = pd.concat(first_parts, ignore_index=True)

    df_full = df_valid.copy()

    print("Compressing repeated (user, template) by last attempt within each period...")
    df_first_last = compress_last_response(df_first)
    df_full_last = compress_last_response(df_full)

    # Templates used in modeling: those appearing in full_last AND present in template_to_skillset
    templates = sorted(list(set(df_full_last["template_id"].unique().tolist()) & set(templates_all)))
    print(f"  Templates used in modeling (from valid users full): {len(templates)}")

    if len(templates) == 0:
        raise ValueError("No templates available after filtering. This should not happen; check data integrity.")

    # Build Q matrix
    print("Building Q matrix...")
    skill_to_index = {s: i for i, s in enumerate(target_skills)}
    Q = np.zeros((len(templates), K), dtype=int)

    for j, tid in enumerate(templates):
        ss = template_to_skillset[tid]  # most_common skill-set
        for s in ss:
            Q[j, skill_to_index[s]] = 1

    # Save Q + template list
    q_df = pd.DataFrame(Q, columns=[f"skill_{s}" for s in target_skills])
    q_df.insert(0, "template_id", templates)
    q_df.to_csv(out / "q_matrix.csv", index=False)
    pd.DataFrame({"template_id": templates}).to_csv(out / "templates.csv", index=False)

    # Build response matrices
    print("Building response matrices R_first / R_full...")
    R_first = build_response_matrix(df_first_last, valid_users, templates)
    R_full = build_response_matrix(df_full_last, valid_users, templates)

    np.save(out / "R_first.npy", R_first)
    np.save(out / "R_full.npy", R_full)

    # Train DINA on FULL
    print("Training DINA on FULL data (slip/guess will be fixed afterwards)...")
    stu_num = R_full.shape[0]
    prob_num = R_full.shape[1]
    know_num = K

    dina = EMDINA(R_full, Q, stu_num, prob_num, know_num, skip_value=-1)
    dina.train(epoch=args.epoch, epsilon=args.epsilon)

    slip = np.asarray(dina.slip, dtype=float)   # (n_templates,)
    guess = np.asarray(dina.guess, dtype=float) # (n_templates,)

    # Save learned item params
    with open(out / "item_params.json", "w") as f:
        json.dump(
            {
                "skills": target_skills,
                "K": K,
                "n_users": int(stu_num),
                "n_templates": int(prob_num),
                "epoch": int(args.epoch),
                "epsilon": float(args.epsilon),
                "templates": templates,
                "slip": slip.tolist(),
                "guess": guess.tolist(),
            },
            f,
            indent=2,
        )

    # Fixed-parameter MAP inference
    print("Inferring MAP mastery (current from first, future from full) with fixed slip/guess...")
    req_masks = templates_to_required_masks(templates, template_to_skillset, skill_to_index)

    alpha_current = infer_map_mastery(R_first, req_masks, slip, guess, K)
    alpha_future = infer_map_mastery(R_full, req_masks, slip, guess, K)

    # Save mastery
    colnames = [f"skill_{s}" for s in target_skills]

    df_cur = pd.DataFrame(alpha_current, columns=colnames)
    df_cur.insert(0, "user_id", valid_users)
    df_cur.to_csv(out / "alpha_current.csv", index=False)

    df_fut = pd.DataFrame(alpha_future, columns=colnames)
    df_fut.insert(0, "user_id", valid_users)
    df_fut.to_csv(out / "alpha_future.csv", index=False)

    # Summary
    summary = {
        "n_users": int(stu_num),
        "n_templates": int(prob_num),
        "K": int(K),
        "skills": target_skills,
        "avg_mastered_current": float(df_cur[colnames].sum(axis=1).mean()),
        "avg_mastered_future": float(df_fut[colnames].sum(axis=1).mean()),
        "avg_net_gain": float((df_fut[colnames].sum(axis=1) - df_cur[colnames].sum(axis=1)).mean()),
        "current_mastery_rate": {str(s): float(df_cur[f"skill_{s}"].mean()) for s in target_skills},
        "future_mastery_rate": {str(s): float(df_fut[f"skill_{s}"].mean()) for s in target_skills},
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Outputs saved to: {out.resolve()}")
    print("Key files:")
    print("  - run_config.json")
    print("  - valid_users.csv")
    print("  - templates.csv")
    print("  - q_matrix.csv")
    print("  - item_params.json (slip/guess)")
    print("  - alpha_current.csv / alpha_future.csv")
    print("  - summary.json")
    print("  - R_first.npy / R_full.npy")


if __name__ == "__main__":
    main()
