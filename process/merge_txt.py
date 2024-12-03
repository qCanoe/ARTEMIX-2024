def read_addresses_from_file(file_path):
    """Read addresses from a file and return them as a set."""
    with open(file_path, 'r') as file:
        addresses = set(line.strip() for line in file)
    return addresses

def write_addresses_to_file(file_path, addresses):
    """Write a set of addresses to a file."""
    with open(file_path, 'w') as file:
        for address in sorted(addresses):
            file.write(f"{address}\n")

def merge_address_files(file_paths, output_file_path):
    """Merge addresses from multiple files and save them to a new file."""
    all_addresses = set()
    for file_path in file_paths:
        addresses = read_addresses_from_file(file_path)
        all_addresses.update(addresses)
    write_addresses_to_file(output_file_path, all_addresses)

if __name__ == "__main__":
    # Replace these with the paths to your actual files
    file_paths = [
        r'qyy\txt\component\1_loop_nodes.txt',
        r'qyy\txt\star_patterns_nodes.txt',
        r'qyy\txt\component\3_short_cycles.txt',
        r'qyy\txt\component\5_isolation_forest.txt',
        r'qyy\txt\component\6_nft_loop.txt'
    ]
    output_file_path = 'qyy/txt/merged_addresses.txt'
    merge_address_files(file_paths, output_file_path)
    print(f"Merged addresses saved to {output_file_path}")
