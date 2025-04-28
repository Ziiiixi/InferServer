import re
import pandas as pd

def parse_log_to_df(log_file):
    """
    Read a log file and return a DataFrame pivoted so that each row is
    (Kernel Name, Kernel Index) and columns are TPC usages with execution times.
    """
    pattern = re.compile(
        r"Name of kernel: (?P<kernel_name>.+?) \| Current iter: \d+ "
        r"\| Client ID: \d+ \| Grid Size: \d+ \| Block Size: \d+ "
        r"\| TPC Usage: (?P<tpc_usage>\d+) \| Critical: \d+ "
        r"\| Kernel Index: (?P<kernel_index>\d+) \| Knee TPC: \d+ "
        r"\| Kernel execution time: (?P<execution_time>\d+) ns"
    )
    rows = []
    with open(log_file, 'r') as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            rows.append({
                'Kernel Name': m.group('kernel_name'),
                'Kernel Index': int(m.group('kernel_index')),
                int(m.group('tpc_usage')): int(m.group('execution_time'))
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=['Kernel Name', 'Kernel Index'],
        values=[c for c in df.columns if isinstance(c, int)],
        aggfunc='first'
    )
    pivot.reset_index(inplace=True)
    pivot.sort_values('Kernel Index', inplace=True)
    return pivot

def compute_knee_tpc(df, threshold=0.05):
    """
    Given a DataFrame with integer-named TPC columns,
    compute the first TPC where latency ≤ (1+threshold)*exclusive_latency.
    Exclusive_latency is assumed at the max TPC column.
    """
    tpc_cols = sorted(c for c in df.columns if isinstance(c, int))
    exclusive = tpc_cols[-2]
    knees = []
    for _, row in df.iterrows():
        base = row[exclusive]
        knee = exclusive
        for tpc in tpc_cols:
            if row[tpc] / base <= 1 + threshold:
                knee = tpc
                break
        knees.append(knee)
    df['Knee TPC'] = knees
    return df

def main():
    log_file  = '123.log'            # Path to your log
    out_csv   = 'kernel_knee_tpc.csv'  # Only CSV output

    df = parse_log_to_df(log_file)
    df = compute_knee_tpc(df, threshold=0.05)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__ == '__main__':
    main()
