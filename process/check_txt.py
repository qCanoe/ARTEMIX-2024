def load_addresses(file_path):
    with open(file_path, 'r') as file:
        addresses = set(line.strip() for line in file if line.strip())
    return addresses

def check_repeated_addresses(file1, file2):
    addresses1 = load_addresses(file1)
    print(len(addresses1))
    addresses2 = load_addresses(file2)
    print(len(addresses2))
    
    repeated_addresses = addresses1.intersection(addresses2)
    unique_addresses1 = addresses1 - repeated_addresses
    unique_addresses2 = addresses2 - repeated_addresses
    
    repeated_count = len(repeated_addresses)
    
    return repeated_count, repeated_addresses, unique_addresses1, unique_addresses2

if __name__ == "__main__":
    file1 = r'qyy\txt\significant_periodic_addresses.txt' 
    
    file2 = r'qyy\txt\True_nodes.txt'  
    
    repeated_count, repeated_addresses, unique_addresses1, unique_addresses2 = check_repeated_addresses(file1, file2)
    
    print(f"Number of unique addresses in the first file: {len(unique_addresses1)}")
    print(f"Number of unique addresses in the second file: {len(unique_addresses2)}")
    
    print(f"Number of repeated addresses: {repeated_count}")

    print(f"{len(repeated_addresses)/(len(unique_addresses1) + len(repeated_addresses)) * 100:.2f}%")    # print(f"Number of unique addresses in the second file: {len(unique_addresses2)}")

        
    
