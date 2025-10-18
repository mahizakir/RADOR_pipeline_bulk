# Rare Disease Oracle (RaDOr) Triple Extraction Pipeline

`run_driver.py` is the driver script to automate the execution of the pipeline_code_with_vllm.py pipeline, which generates simplified sentences and structured triples from rare disease literature. The outputs will be used later for building knowledge graphs.



# Overview

The pipeline processes disease-specific text data stored in the folders:

`results_after_year_2000/<disease_name>/`

For each disease, it generates two output CSV files and stores them inside the same disease specific data folder:

- `generated_simple_sentences_<disease_name>.csv` — contains simplified sentences extracted from the text.
- `triples_<disease_name>.csv` — contains structured triples derived from the simplified sentences.

The goal is to process all 5000+ rare diseases found inside the `results_after_year_2000` folder.




  

# Indexing the number of disease folders for run: 
To manage large-scale processing, diseases should be run in ranges because we cannot process 5000+ disease data at once (GPU Time for 1 diseases is between 6 to 200+ minutes):

**Example for partial batch run**
Inside the run_driver.py file:

- `start_index = 1000`  # starts at the 1000th disease (eg. start_index = 11 processes 12th row of csv)
- `end_index = 1100`    # (eg. end_index = 15 stop processing after processing 15th row of csv)

- Full dataset inside results_after_year_2000 folder has 5094 diseases.


# Typical GPU runtime:
  - ~20–25 diseases per day on high-end GPUs (e.g., A6000).
  - Running disease from index 1000–1100 may take ~5 days.
- You can adjust start_index and end_index to split the workload across multiple sessions or machines.



# Activate Python environment
Make one following requirements --> must need vllm, torch, pandas, csv, sentence-transformers, nltk, tiktoken and more.

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
Adjust the start_index and end_index for full run or uncomment the test block for testing.

`python run_driver.py`

The script will sequentially run the pipeline for each disease in the selected range.

# Output log
Main output log is found in `DRAC_output_log_from_pipeline.log` (ignore the prints in terminal)**

