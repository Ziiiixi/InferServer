import pandas as pd

# Load the first file into a DataFrame
file1_path = 'resnet152_32_fwd_updated'  # Replace with your file path
file2_path = 'kernel_performance_with_Knee_TPC_and_Is_Critical.csv'  # Replace with your file path
output_path = 'updated_file1.csv'  # Path to save the updated file

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Ensure the column names match your files
df2 = df2.rename(columns={"Kernel Name": "Name", "Grid Size": "Grid", "Block Size": "Block"})

# Assign Knee_TPC values based on Name, Grid, and Block Size
for index, row in df1.iterrows():
    match = df2[
        (df2["Name"] == row["Name"]) 
        # &
        # (df2["Grid"] == row["Grid"]) &
        # (df2["Block"] == row["Block"])
    ]
    if not match.empty:
        
        df1.at[index, "Knee_TPC"] = match["Knee_TPC"].values[0]
        df1.at[index, "Is_Critical"] = match["Is_Critical"].values[0]

# Save the updated DataFrame to a new file
df1.to_csv(output_path, index=False)
print(f"Updated file saved to {output_path}")
