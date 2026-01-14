from pathlib import Path

# Load all Training Set B UIDs
trainingset_b_file = 'trainingsetB_uids.txt'
with open(trainingset_b_file, 'r') as f:
    trainingset_b_uids = set(line.strip() for line in f if line.strip())

# Load already-downloaded UIDs
high_quality_file = 'high_quality_uids.txt'
with open(high_quality_file, 'r') as f:
    high_quality_uids = set(line.strip() for line in f if line.strip())

print(f"Training Set B UIDs: {len(trainingset_b_uids):,}")
print(f"Already downloaded (high_quality): {len(high_quality_uids):,}")


# Filter out already-downloaded UIDs
remaining_uids = trainingset_b_uids - high_quality_uids
remaining_uids = sorted(remaining_uids)  # Sort for consistency

print(f"Remaining UIDs to download: {len(remaining_uids):,}")
print(f"Overlap (already downloaded): {len(trainingset_b_uids & high_quality_uids):,}")


# Split into chunks of max 30k UIDs
CHUNK_SIZE = 30000

num_chunks = (len(remaining_uids) + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"Splitting {len(remaining_uids):,} UIDs into {num_chunks} files (max {CHUNK_SIZE:,} per file)\n")

for i in range(num_chunks):
    start_idx = i * CHUNK_SIZE
    end_idx = min((i + 1) * CHUNK_SIZE, len(remaining_uids))
    chunk = remaining_uids[start_idx:end_idx]
    
    output_file = f'trainingsetB_part{i+1}.txt'
    with open(output_file, 'w') as f:
        for uid in chunk:
            f.write(f"{uid}\n")
    
    print(f"Saved {len(chunk):,} UIDs to '{output_file}'")

print(f"\nTotal: {len(remaining_uids):,} UIDs saved across {num_chunks} files")