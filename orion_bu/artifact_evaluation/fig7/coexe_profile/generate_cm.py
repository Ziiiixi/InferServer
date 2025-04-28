import pandas as pd
import re
import os
# Load the CSVs (assuming the files are in the same directory)
def load_kernel_info(model_name1, model_name2):
    filename = ''
    if(model_name1 == model_name2):
        filename = f'/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/kernel_groups/{model_name1}_groups.csv'
    
    else:
        filename = f'/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/kernel_groups/{model_name1}_{model_name2}_groups.csv'
        
        if not os.path.exists(filename):
            # Check if the reversed filename exists
            filename = f'/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/kernel_groups/{model_name2}_{model_name1}_groups.csv'
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Neither {model_name1}_{model_name2}_groups.csv nor {model_name2}_{model_name1}_groups.csv found.")
            
    return pd.read_csv(filename)


def load_contention_matrix(model_name1, model_name2):
    filename = ''
    if(model_name1 == model_name2):
        filename = f'/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/contention_matrix/contention_matrix_{model_name1}.csv'
    
    else:
        filename = f'/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/contention_matrix/contention_matrix_{model_name1}_{model_name2}.csv'
        if not os.path.exists(filename):
            # Check if the reversed filename exists
            filename = f'/home/zixi/orion_bu/artifact_evaluation/fig7/coexe_profile/contention_matrix/contention_matrix_{model_name2}_{model_name1}.csv'
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Neither contention_matrix_{model_name1}_{model_name2}.csv nor contention_matrix_{model_name2}_{model_name1}.csv found.")
    
    return pd.read_csv(filename)

# Function to get the group based on kernel ID
def get_kernel_group(kernel_id, kernel_info_df):
    """Find the group (cluster) based on the kernel ID."""
    kernel_info = kernel_info_df[kernel_info_df['Kernel_ID'] == kernel_id]
    if kernel_info.empty:
        return None
    return kernel_info.iloc[0]['Cluster']

# Function to get the contention factor between two groups
def get_contention_factor(group_1, group_2, contention_matrix_df):
    """Get the contention factor between two groups."""
    result = contention_matrix_df[(contention_matrix_df['Group_1'] == group_1) & 
                                  (contention_matrix_df['Group_2'] == group_2)]
    if result.empty:
        return None
    return result.iloc[0]['Contention_Factor']

# Function to read and parse the profile plan from a log file using pattern matching
def read_profile_plan_from_log(file_path):
    """Read and parse the kernel pairs from a log file."""
    profile_plan = []
    
    # Regular expression pattern for extracting Kernel pairs and Model information
    pattern = re.compile(r"Kernel_ID_1\s*=\s*(\d+),\s*Kernel_ID_2\s*=\s*(\d+),\s*Model_1\s*=\s*(\S+),\s*Model_2\s*=\s*(\S+),\s*Cluster_1\s*=\s*(\d+),\s*Cluster_2\s*=\s*(\d+)")
    
    # Open the log file and search for matches
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Extract the kernel pair information
                kernel_1_id = int(match.group(1))
                kernel_2_id = int(match.group(2))
                model_1 = match.group(3)
                model_2 = match.group(4)
                cluster_1 = int(match.group(5))
                cluster_2 = int(match.group(6))
                profile_plan.append((kernel_1_id, kernel_2_id, model_1, model_2, cluster_1, cluster_2))
    
    return profile_plan

def update_contention_matrix(data, file_path):
    """Update the contention matrix CSV with new data."""
    # Load the existing CSV into a DataFrame
    contention_matrix_df = pd.read_csv(file_path)
    
    # Loop through the data and update the corresponding entries
    for entry in data:
        group_1, group_2, contention_factor = entry
        
        # Update the rows with matching group_1 and group_2 values
        contention_matrix_df.loc[(contention_matrix_df['Group_1'] == group_1) & 
                                  (contention_matrix_df['Group_2'] == group_2), 'Contention_Factor'] = contention_factor
        # Update the reverse pair as well (group_2, group_1)
        contention_matrix_df.loc[(contention_matrix_df['Group_1'] == group_2) & 
                                  (contention_matrix_df['Group_2'] == group_1), 'Contention_Factor'] = contention_factor
    
    # Save the updated DataFrame back to the CSV
    contention_matrix_df.to_csv(file_path, index=False)
    print(f"Contention factors updated in '{file_path}'.")

# Read the profile plan from the log file
log_file_path = '/home/zixi/orion_bu/artifact_evaluation/fig7/123.log'  # Adjust the path to your log file
profile_plan = read_profile_plan_from_log(log_file_path)

# Initialize a list to store valid entries to be updated in the CSV
data_to_update = []

# Process each pair of kernels (now only models are the same)
for kernel_1_id, kernel_2_id, model_1, model_2, cluster_1, cluster_2 in profile_plan:
    # Load the kernel group information
    kernel_info = load_kernel_info(model_1, model_2)
    
    # Get the group (cluster) of each kernel
    group_1 = get_kernel_group(kernel_1_id, kernel_info)
    group_2 = get_kernel_group(kernel_2_id, kernel_info)
    
    if group_1 is None or group_2 is None:
        print(f"Kernel ID(s) {kernel_1_id} or {kernel_2_id} not found in {model_1} or {model_2} group file.")
        continue
    
    contention_matrix = load_contention_matrix(model_1, model_2)
    contention_factor = get_contention_factor(group_1, group_2, contention_matrix)
    
    data_to_update.append([cluster_1, cluster_2, contention_factor])  # Original order
    data_to_update.append([cluster_2, cluster_1, contention_factor])  # Reversed order

# After processing all pairs, update the data in the CSV
if data_to_update:
    update_contention_matrix(data_to_update, 'contention_matrix.csv')
else:
    print("No contention factors with value 0 found.")
