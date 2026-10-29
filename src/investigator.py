"""
investigator.py
---------------
Alert Investigator: turns thousands of flagged 15-minute intervals into a short,
ranked list of "incidents" with a plain-English explanation.

Pipeline (observe -> group -> investigate -> decide -> report)
--------------------------------------------------------------
1. OBSERVE     take the intervals the RBM flagged (score > threshold)
2. GROUP       merge flags that are close together into one incident
3. INVESTIGATE compare each incident with the training-period median for the
               same hours of the day, feature by feature
4. DECIDE      give each incident a descriptive label and a severity rank
5. REPORT      write a digest; optionally ask an LLM to rewrite it more nicely

Everything up to step 5 is plain rules (no AI model, works offline).
The LLM step is OPTIONAL. It only receives the findings produced above and is
told not to add facts or claim faults. If it fails, the rule-based digest is used.

Labels are DESCRIPTIVE ("low-usage period"), never fault diagnoses.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_dotenv(path: Optional[Path] = None) -> None:
    """Load KEY=value lines from a .env file in the project root (no extra package needed).
    Real environment variables win over the file."""
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

INTERVAL_MIN = 15
FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
FRIENDLY = {
    "Global_active_power": "active power",
    "Global_reactive_power": "reactive power",
    "Voltage": "voltage",
    "Global_intensity": "current",
    "Sub_metering_1": "kitchen circuit",
    "Sub_metering_2": "laundry circuit",
    "Sub_metering_3": "water heater/AC circuit",
}
UNITS = {
    "Global_active_power": "kW",
    "Global_reactive_power": "kW",
    "Voltage": "V",
    "Global_intensity": "A",
    "Sub_metering_1": "Wh per 15 min",
    "Sub_metering_2": "Wh per 15 min",
    "Sub_metering_3": "Wh per 15 min",
}
DISCLAIMER = (
    "These are statistically unusual patterns compared with the training period. "
    "They are NOT confirmed appliance faults; a person must check them."
)


# --------------------------------------------------------------------------- #
# 1-2. Group flagged intervals into incidents
# --------------------------------------------------------------------------- #
def group_incidents(
    scores: np.ndarray,
    threshold: float,
    index: pd.DatetimeIndex,
    max_gap: int = 2,
) -> List[dict]:
    """
    Merge flagged intervals into incidents.

    Two flagged intervals belong to the same incident if they are at most
    `max_gap` positions apart (default 2 = within 30 minutes).
    """
    flagged = np.where(np.asarray(scores) > threshold)[0]
    if len(flagged) == 0:
        return []

    groups, current = [], [flagged[0]]
    for i in flagged[1:]:
        if i - current[-1] <= max_gap:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    incidents = []
    for n, g in enumerate(groups, 1):
        first, last = g[0], g[-1]
        sc = np.asarray(scores)[g]
        peak_pos = g[int(np.argmax(sc))]
        incidents.append({
            "id": n,
            "first_pos": int(first),
            "last_pos": int(last),
            "start": index[first],
            "end": index[last] + pd.Timedelta(minutes=INTERVAL_MIN),
            "span_intervals": int(last - first + 1),
            "n_flagged": int(len(g)),
            "peak_score": float(sc.max()),
            "mean_score": float(sc.mean()),
            "peak_time": index[peak_pos],
        })
    return incidents


# --------------------------------------------------------------------------- #
# 3. Investigate one incident
# --------------------------------------------------------------------------- #
def hourly_typical(train_df: pd.DataFrame) -> pd.DataFrame:
    """Median of every feature for each hour of the day, from training data."""
    cols = [c for c in FEATURES if c in train_df.columns]
    return train_df[cols].groupby(train_df.index.hour).median()


def investigate_incident(incident: dict, df: pd.DataFrame, typical_by_hour: pd.DataFrame) -> dict:
    """
    Compare the readings during an incident with the typical readings for the
    same hours of the day. Adds 'findings' (per-feature comparison) to the incident.
    """
    seg = df.iloc[incident["first_pos"]: incident["last_pos"] + 1]
    hours = seg.index.hour
    findings = {}
    for col in typical_by_hour.columns:
        if col not in seg.columns:
            continue
        actual = float(np.nanmean(seg[col].values)) if seg[col].notna().any() else np.nan
        typical = float(np.nanmean(typical_by_hour.loc[hours, col].values))
        if np.isnan(actual) or np.isnan(typical):
            continue
        ratio = actual / typical if abs(typical) > 1e-6 else np.nan
        findings[col] = {"actual": actual, "typical": typical, "ratio": ratio}

    out = dict(incident)
    out["findings"] = findings
    out["hours"] = sorted(set(int(h) for h in hours))
    out["weekday"] = incident["start"].strftime("%a")
    return out


# --------------------------------------------------------------------------- #
# 4. Decide: label + severity
# --------------------------------------------------------------------------- #
def label_incident(inc: dict) -> str:
    """Descriptive (non-diagnostic) label chosen by simple rules, in priority order."""
    f = inc.get("findings", {})
    gap = f.get("Global_active_power")
    night = all(h <= 5 for h in inc.get("hours", [12])) if inc.get("hours") else False

    if gap and not np.isnan(gap["ratio"]):
        if night and gap["ratio"] > 1.5:
            return "Overnight usage above normal"
        if gap["ratio"] < 0.5:
            return "Low-usage period"
        if gap["ratio"] > 1.5:
            return "High-usage period"

    for col in ("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"):
        s = f.get(col)
        if s and ((not np.isnan(s["ratio"]) and s["ratio"] > 2.0) or (s["typical"] < 0.5 and s["actual"] > 2.0)):
            return f"Unusual {FRIENDLY[col]} activity"

    v = f.get("Voltage")
    if v and not np.isnan(v["ratio"]) and abs(v["ratio"] - 1) > 0.02:
        return "Voltage away from usual level"

    return "Unusual combination of readings"


def rank_incidents(incidents: List[dict], val_scores: np.ndarray, threshold: float) -> List[dict]:
    """
    Add a severity score and level.

    severity = how far the peak is above the threshold (in validation-score
    standard deviations) + a small bonus for longer incidents.
    Levels: top 10% High, next 30% Medium, rest Low.
    """
    std = float(np.std(val_scores)) or 1.0
    for inc in incidents:
        inc["severity"] = (inc["peak_score"] - threshold) / std + 0.3 * np.log1p(inc["n_flagged"])
    if not incidents:
        return incidents
    sev = np.array([i["severity"] for i in incidents])
    hi, mid = np.quantile(sev, 0.90), np.quantile(sev, 0.60)
    for inc in incidents:
        inc["level"] = "High" if inc["severity"] >= hi else ("Medium" if inc["severity"] >= mid else "Low")
    return sorted(incidents, key=lambda i: i["severity"], reverse=True)


def run_investigation(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    train_df: pd.DataFrame,
    val_scores: np.ndarray,
    max_gap: int = 2,
) -> List[dict]:
    """Full agent run: group -> investigate -> label -> rank. Returns incidents, most severe first."""
    incs = group_incidents(scores, threshold, df.index, max_gap=max_gap)
    typical = hourly_typical(train_df)
    out = []
    for inc in incs:
        inc = investigate_incident(inc, df, typical)
        inc["label"] = label_incident(inc)
        out.append(inc)
    return rank_incidents(out, val_scores, threshold)


# --------------------------------------------------------------------------- #
# 5. Report
# --------------------------------------------------------------------------- #
def _top_deviations(inc: dict, k: int = 3) -> List[str]:
    items = []
    for col, s in inc["findings"].items():
        if np.isnan(s["ratio"]):
            if s["actual"] > 2.0:   # active while normally idle (sub-meters)
                items.append((9.0, f"{FRIENDLY[col]} {s['actual']:.1f} {UNITS[col]} vs ~0 typical"))
            continue
        u = UNITS[col]
        items.append((abs(np.log(max(s["ratio"], 1e-3))),
                      f"{FRIENDLY[col]} {s['actual']:.2f} {u} vs {s['typical']:.2f} {u} typical "
                      f"({(s['ratio'] - 1) * 100:+.0f}%)"))
    items.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in items[:k]]


def incident_facts(inc: dict) -> str:
    """One incident as a compact text block (also the only input the LLM sees)."""
    dur_h = inc["span_intervals"] * INTERVAL_MIN / 60
    return (
        f"Incident #{inc['id']} [{inc['level']} severity] {inc['weekday']} "
        f"{inc['start']:%Y-%m-%d %H:%M} to {inc['end']:%H:%M} "
        f"({dur_h:.2g} h span, {inc['n_flagged']} flagged intervals). "
        f"Label: {inc['label']}. "
        f"Largest differences from typical for these hours: {'; '.join(_top_deviations(inc)) or 'none'}."
    )


def build_digest(incidents: List[dict], total_flagged: int, top_n: int = 5) -> str:
    """Rule-based plain-text alert digest (works offline)."""
    lines = [
        "ALERT DIGEST",
        f"{total_flagged} flagged intervals were grouped into {len(incidents)} incidents. "
        f"Top {min(top_n, len(incidents))} by severity:",
        "",
    ]
    for inc in incidents[:top_n]:
        lines.append("- " + incident_facts(inc))
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional LLM layer (OpenAI-compatible API, e.g. Groq)
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"   # change if your Groq console lists another model

SYSTEM_PROMPT = (
    "You write short alert summaries for a household energy-monitoring dashboard. "
    "Use ONLY the facts you are given. Do not invent causes, appliances, places or numbers. "
    "Copy every number WITH the unit exactly as given (kW, Wh per 15 min, V, A); never convert units "
    "and never write kWh. "
    "Never say an appliance is faulty or broken; say 'unusual pattern' and suggest a quick human check. "
    "Write plain English for a non-technical homeowner: at most 50 words per incident, "
    "one short paragraph each, then one closing sentence. Finish every incident you are given."
)


def _trim_to_last_sentence(text: str) -> str:
    """If the reply was cut off mid-sentence, drop the unfinished part."""
    idx = max(text.rfind(c) for c in ".!?")
    if idx > 40:
        return text[: idx + 1] + "\n[Summary shortened: the AI reply was cut off.]"
    return text


def llm_summary(
    facts: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    Call an OpenAI-compatible /chat/completions endpoint and return the text. Raises on any failure.

    Some models spend part of max_tokens on hidden reasoning, which can cut the visible answer
    short (finish_reason == "length"). In that case we retry once with a bigger limit, and if it
    is still cut off we trim the unfinished sentence.
    """
    base = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    mdl = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    text = ""
    for max_tokens in (3000, 5000):
        body = json.dumps({
            "model": mdl,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Summarise these findings:\n\n" + facts},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            base + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "appliance-doctor/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = data["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if choice.get("finish_reason") != "length" and text:
            return text
    return _trim_to_last_sentence(text) if text else text


def summarise_with_fallback(
    incidents: List[dict],
    total_flagged: int,
    use_llm: bool,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    top_n: int = 5,
    llm_fn=llm_summary,
) -> Tuple[str, bool, Optional[str]]:
    """
    Return (text, used_llm, error_message).
    Falls back to the rule-based digest if the LLM is off, has no key, or fails.
    """
    digest = build_digest(incidents, total_flagged, top_n)
    if not use_llm:
        return digest, False, None
    key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        return digest, False, "No API key set - showing the rule-based digest."
    facts = "\n".join(incident_facts(i) for i in incidents[:top_n])
    try:
        text = llm_fn(facts, key, base_url, model)
        return text + "\n\n" + DISCLAIMER, True, None
    except urllib.error.HTTPError as e:
        return digest, False, f"AI summary failed (HTTP {e.code}) - showing the rule-based digest."
    except Exception as e:  # network, timeout, bad JSON ...
        return digest, False, f"AI summary failed ({type(e).__name__}) - showing the rule-based digest."