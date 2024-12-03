import pandas as pd

# File paths
true_positive_nodes_path = r"qyy\txt\final_suspicious_addresses_boosting.txt"
embedding_csv_path = "original_data/每个地址被标记wash的次数.csv"
output_path = "qyy/txt/final_suspicious_addresses_boosting.txt"

# Read the true positive nodes
with open(true_positive_nodes_path, 'r') as file:
    indices = [int(line.strip()) for line in file]

# Read the embedding CSV without header and take only the first column
embedding_df = pd.read_csv(embedding_csv_path, header=None, usecols=[0])

# Ensure indices are within the bounds of the CSV
max_index = len(embedding_df)
indices = [i for i in indices if i < max_index]

# Select the corresponding rows from the embedding CSV
selected_rows = embedding_df.iloc[indices]

# Save the selected rows to a new text file, each address on a separate line
with open(output_path, 'w') as output_file:
    for address in selected_rows[0]:
        output_file.write(f"{address}\n")

print(f"Selected addresses have been saved to {output_path}")
