#!/usr/bin/env python3
import os
import re
import pandas as pd

# ─── EDIT THESE ────────────────────────────────────────────────────────────────
FIRST_CSV    = "densenet201_8_fwd.csv"                        # your input file
NCU_CSV      = "~/orion_bu/profiling/benchmarks/resnet152_bz8/output_ncu_sms_roofline.csv"        # NCU roofline file
UPDATED_CSV  = "densenet201_8_fwd_updated.csv"                # where to write the result
# ──────────────────────────────────────────────────────────────────────────────

def update_critical_ops(first_csv_path, second_csv_path, output_path):
    # 1) Read the two CSVs
    df1 = pd.read_csv(first_csv_path)
    df2 = pd.read_csv(second_csv_path)

    # 2) Filter out unwanted kernels in the NCU file
    df2 = df2[~df2['Kernel_Name']
                 .str.contains('splitKreduce_kernel|reduce_kernel',
                               case=False, na=False)].copy()
    df2['Kernel_Name'] = df2['Kernel_Name'].str.strip('"')

    # 3) Normalize Grid/Block to strings for exact matching
    for df in (df1, df2):
        df['Grid']  = df['Grid'].astype(str)
        df['Block'] = df['Block'].astype(str)

    # 4) For each row in df1, check df2 for Compute(SM)(%) > 50
    def compute_is_critical(row):
        name_pat = re.escape(row['Name'])
        matches = df2[
            df2['Kernel_Name'].str.contains(name_pat, regex=True, na=False) &
            (df2['Grid']  == row['Grid']) &
            (df2['Block'] == row['Block'])
        ]
        return int((matches['Compute(SM)(%)'] > 50).any())

    df1['is critical'] = df1.apply(compute_is_critical, axis=1)

    # 5) Write out the updated CSV
    df1.to_csv(output_path, index=False)
    print(f"✓ Written updated CSV to: {output_path}")

if __name__ == "__main__":
    update_critical_ops(FIRST_CSV, NCU_CSV, UPDATED_CSV)
