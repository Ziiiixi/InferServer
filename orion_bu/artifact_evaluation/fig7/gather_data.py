#!/usr/bin/env python3
# gather_data.py
"""
Parse a log that reports, for each Client <i>,
    … Client <i> … p99: <sec> … average: <sec> …
Group consecutive records so each client appears once per group,
then emit two summary rows (p99 / avg) across all groups.
"""

import re
import pandas as pd
from math import nan  # placeholder for missing metrics

LOG_FILE     = "123.log"   # ← change if needed
NUM_CLIENTS  = 4           # expecting client IDs 0 … 3

# --- regex to extract client id, p99 latency and average latency ------------
pat = re.compile(
    r"Client\s+(?P<cid>\d+).*?p99:\s+(?P<p99>[0-9.]+).*?average:\s+(?P<avg>[0-9.]+)",
    re.IGNORECASE,
)

# --- helper -----------------------------------------------------------------
def flush_group(buf, gidx):
    """Return one dict for the current group & clear the buffer."""
    row = {"group": gidx}
    for c in range(NUM_CLIENTS):
        row[f"p99_client{c}"] = buf.get(c, {}).get("p99_sec", nan)
        row[f"avg_client{c}"] = buf.get(c, {}).get("avg_sec", nan)
    return row

# --- main parsing loop ------------------------------------------------------
groups, buffer, gidx = [], {}, 1

with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue

        cid = int(m["cid"])         # e.g. 0-3
        metrics = {
            "p99_sec": float(m["p99"]),
            "avg_sec": float(m["avg"]),
        }

        # already saw this client → start a new group
        if cid in buffer:
            groups.append(flush_group(buffer, gidx))
            gidx  += 1
            buffer = {}

        buffer[cid] = metrics

        # once we have all clients, close the group
        if len(buffer) == NUM_CLIENTS:
            groups.append(flush_group(buffer, gidx))
            gidx  += 1
            buffer = {}

# flush any trailing partial group
if buffer:
    groups.append(flush_group(buffer, gidx))

# --- build DataFrame --------------------------------------------------------
cols = (["group"] +
        [f"p99_client{c}" for c in range(NUM_CLIENTS)] +
        [f"avg_client{c}" for c in range(NUM_CLIENTS)])
df = pd.DataFrame(groups, columns=cols)

# --- create two summary rows (p99 / avg) ------------------------------------
p99_row, avg_row = [], []
for _, row in df.iterrows():
    for c in range(NUM_CLIENTS):
        p99_row.append(row[f"p99_client{c}"])
    for c in range(NUM_CLIENTS):
        avg_row.append(row[f"avg_client{c}"])

final_df = pd.DataFrame([p99_row, avg_row])

print(final_df.to_string(index=False))
final_df.to_csv("latency_groups.csv", index=False, header=False)
