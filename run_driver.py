# driver_run_pipeline.py

import subprocess
import csv

csv_path = "static_data/5094_disease_folders_names.csv"
# Read CSV and extract names
with open(csv_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    all_diseases = [row[0] for row in reader if row]  # ignore empty lines

# Choose range (for example, from 0 till 5094 at max)
# For now, we want to run from index 1000 to 1100 this should take 5 days gpu time. Usually 20-25 diseases per day.
# Overall there are 5094 diseases in the csv. So we start from 1000 in DRAC and go till 5094 in total, 
# I will run the first from 0 to 1000 in Labs A6000.
# Adjust these indices as needed

start_index = 1000 #  note: 11 processes 12th row of csv
end_index = 1100 # note: 15 means it will stop after processing the 15th row.

disease_names = all_diseases[start_index:end_index]
print("Driver will generate triples for ", len(disease_names), "diseases.")  

print(f"Selected {len(disease_names)} diseases (rows {start_index} to {end_index-1}):")


# TEST CASE: List of disease names as example
disease_names = [
    "1q44_microdeletion_syndrome",
    "GNE_myopathy",
]

# Path to your pipeline script
pipeline_script = "pipeline_code_with_vllm.py"

for disease in disease_names:
    print(f"\n=== Running pipeline for: {disease} ===\n")
    # Run the pipeline script as a subprocess and pass disease_name as an argument
    subprocess.run(["python3", pipeline_script, disease], check=True)
