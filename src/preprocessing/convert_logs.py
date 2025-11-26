import pandas as pd
import os
import glob

RAW_DIR = "data/raw"
OUT_DIR = "data/intermediate"

os.makedirs(OUT_DIR, exist_ok=True)

def convert_zeek_log_to_csv(input_file, output_file):
    # Load Zeek log 
    df = pd.read_csv(
        input_file,
        sep="\t",
        comment="#",
        header=None,
        low_memory=False
    )

    # Extract header from Zeek metadata
    with open(input_file, "r") as f:
        for line in f:
            if line.startswith("#fields"):
                columns = line.strip().split("\t")[1:]
                break

    df.columns = columns
    df.to_csv(output_file, index=False)
    print(f"Saved → {output_file}")

# Process all scenarios
log_files = glob.glob(f"{RAW_DIR}/**/conn.log.labeled", recursive=True)
print("Found", len(log_files), "log files.")

for file in log_files:
    scenario = file.split("/")[-2]
    out_csv = f"{OUT_DIR}/{scenario}.csv"
    convert_zeek_log_to_csv(file, out_csv)
