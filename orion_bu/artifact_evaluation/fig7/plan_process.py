import csv
import re

# Define the path to your log file
log_file_path = '123.log'

# Step 1: Parse the log file to extract the kernel pairs and clusters
kernel_pairs = []
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        match = re.match(r"Pair: Kernel_ID_1 = (\d+), Kernel_ID_2 = (\d+), .*Cluster_1 = (\d+), Cluster_2 = (\d+)", line)
        if match:
            kernel_pairs.append((int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))))

# Step 2: Initialize the contention matrix (contention factors set to 0 initially)
# We'll use a list of tuples to maintain the order
contention_matrix = []

for kernel_id_1, kernel_id_2, cluster_1, cluster_2 in kernel_pairs:
    # Check if the pair already exists in the matrix (avoid duplicates)
    if not any((group_1 == cluster_1 and group_2 == cluster_2) or (group_1 == cluster_2 and group_2 == cluster_1) for group_1, group_2, _ in contention_matrix):
        contention_matrix.append((cluster_1, cluster_2, 0))

# Step 3: Write the matrix to a CSV file, maintaining the order from the log
csv_filename = 'contention_matrix.csv'
header = ['Group_1', 'Group_2', 'Contention_Factor']

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Write the contention matrix data
    for group_1, group_2, contention_factor in contention_matrix:
        writer.writerow([group_1, group_2, contention_factor])

print(f'Contention matrix CSV file "{csv_filename}" has been created successfully.')
