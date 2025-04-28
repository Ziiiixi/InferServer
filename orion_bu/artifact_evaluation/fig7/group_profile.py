import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# --- User config: where to write the per-model groupings ---
groups_output_dir = '/home/zixi/orion_bu/artifact_evaluation/fig7/'
os.makedirs(groups_output_dir, exist_ok=True)


trace_files = [
    ("", "", "Tnet_8", 160000),
    # ("", "", "Rnet_8_R1net_8", 160000),  # example combined run
]

base_dir = '/home/zixi/orion_bu/profiling/benchmarks/'
exe_time_dir = '/home/zixi/orion_bu/artifact_evaluation/fig7/kernel_profiles/'
model_name_mapping = {
    "Mnet_32": "mobilenetv2_bz32",
    "Rnet_32": "resnet152_bz32",
    "R1net_32": "resnet101_bz32",
    "Dnet_32": "densenet201_bz32",
    "Vnet_32": "vgg19_bz32",
    "Rnet_16": "resnet152_bz16",
    "Dnet_16": "densenet201_bz16",
    "Rnet_8":  "resnet152_bz8",
    "R1net_8": "resnet101_bz8",
    "Dnet_8": "densenet201_bz8",
    "Vnet_8": "vgg19_bz8",
    "Tnet_8": "vitb16_bz8",
    # add others as needed
}


for _, _, trace_model, _ in trace_files:
    print(f"\nProcessing trace_model = {trace_model!r}")
    # split into sub‑models: groups of two tokens, e.g. ["Rnet","8","R1net","8"] → ["Rnet_8","R1net_8"]
    tokens     = trace_model.split('_')
    submodels  = ['_'.join(tokens[i:i+2]) for i in range(0, len(tokens), 2)]
    
    dfs = []
    for model_name in submodels:
        if model_name not in model_name_mapping:
            print(f"  ⚠️  Skipping unknown model {model_name}")
            continue

        mapped = model_name_mapping[model_name]
        trace_fp = os.path.join(base_dir, f'{mapped}/output_ncu_sms_roofline.csv')
        time_fp  = os.path.join(exe_time_dir, f'{mapped}.csv')
        
        if not (os.path.exists(trace_fp) and os.path.exists(time_fp)):
            print(f"  ⚠️  Missing files for {model_name}:")
            print(f"      {trace_fp}")
            print(f"      {time_fp}")
            continue
        
        df = pd.read_csv(trace_fp)
        # filter out unwanted kernels
        df = df[~df['Kernel_Name']
                  .str.contains('splitKreduce_kernel|reduce_kernel', case=False, na=False)]
        df = df.reset_index(drop=True)
        df['Kernel_ID'] = df.index
        df['Model']     = model_name
        
        exec_df = pd.read_csv(time_fp, usecols=['Kernel Index','12'])
        df = df.merge(exec_df,
                      left_on='Kernel_ID',
                      right_on='Kernel Index',
                      how='left') \
               .drop(columns=['Kernel Index']) \
               .rename(columns={'12':'Execution_Time'})
        
        dfs.append(df)
        print(f"  ✓ Loaded {model_name}: {len(df)} rows")
    
    if not dfs:
        print(f"  ⚠️  No data for {trace_model}, skipping clustering.")
        continue

    # --- combine, preprocess & cluster once for this trace_model ---
    merged = pd.concat(dfs, ignore_index=True)
    # fix AI feature
    merged['AI(flops/bytes)'] = merged['AI(flops/bytes)'] \
        .replace([-1,0], np.nan) \
        .fillna(merged['AI(flops/bytes)'].median())

    features = ['DRAM_Throughput(%)','Compute(SM)(%)',
                'Registers_Per_Thread','Number_of_threads',
                'AI(flops/bytes)']
    X = StandardScaler().fit_transform(merged[features].values)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.5,
        linkage='ward'
    )
    merged['Cluster'] = clustering.fit_predict(X)

    # reorder so Kernel_ID is first
    cols = ['Kernel_ID'] + [c for c in merged.columns if c != 'Kernel_ID']
    merged = merged[cols]

    # --- write out one file per trace_model ---
    out_path = os.path.join(groups_output_dir, f"{trace_model}_groups.csv")
    merged.to_csv(out_path, index=False)
    print(f"  → Wrote clustering for {trace_model} to {out_path}")
