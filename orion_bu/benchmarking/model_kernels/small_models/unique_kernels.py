import pandas as pd

def count_unique_kernels(csv_path):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Ensure Grid and Block are treated as strings (in case they look numeric)
    df['Grid'] = df['Grid'].astype(str)
    df['Block'] = df['Block'].astype(str)
    
    # Drop duplicates based on Name, Grid, Block
    unique_kernels = df.drop_duplicates(subset=['Name', 'Grid', 'Block'])
    
    # Count them
    count = unique_kernels.shape[0]
    
    # Optionally, get the list of unique (Name, Grid, Block) tuples
    kernel_list = list(unique_kernels[['Name', 'Grid', 'Block']].itertuples(index=False, name=None))
    
    return count, kernel_list

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Count unique kernels in a CSV.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()
    
    count, kernels = count_unique_kernels(args.csv_file)
    
    print(f'Found {count} unique kernels based on (Name, Grid, Block):\n')
    for name, grid, block in kernels:
        print(f'  - {name} | Grid={grid} | Block={block}')
