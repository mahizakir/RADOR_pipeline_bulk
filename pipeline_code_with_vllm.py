import os, sys
import pandas as pd
import re
import tiktoken
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import time
start_time = time.time() 

import sys

# Function to print and append only selected messages
def log_to_file(*args, file_path="DRAC_output_log_from_pipeline.log"):
    with open(file_path, "a") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(*args)  # handle multiple arguments like print()
        sys.stdout = old_stdout


# Settings
base_dir = "results_after_year_2000"
disease_name = sys.argv[1]
csv_file = os.path.join(base_dir, disease_name, "output_final_file_selection.csv")
fulltexts_dir = os.path.join(base_dir, disease_name, "fulltexts")
log_to_file("Processing disease: " + disease_name)

# Tokenizer (OpenAI embeddings)
enc = tiktoken.get_encoding("cl100k_base")
max_tokens = 300  # max tokens per chunk

# Load CSV
df = pd.read_csv(csv_file)
# df = pd.read_csv(csv_file, nrows=1) # do this if you want to test with just one article


chunks_list = []
metadata_list = []
sizes_list = []

for _, row in df.iterrows():
    fname = row["Filename"]
    txt_path = os.path.join(fulltexts_dir, fname)
    if not txt_path.endswith(".txt"):
        txt_path += ".txt"

    if not os.path.exists(txt_path):
        print(f"⚠️ File missing: {txt_path}")
        continue

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    meta = ". ".join([
        str(row["title"]),
        str(row["authors"]),
        str(row["journal"]),
        str(row["pubdate"]),
        str(row["pmc_id"])
    ])

    for para in paragraphs:
        para_tokens = enc.encode(para)
        if len(para_tokens) <= max_tokens:
            # Keep paragraph as single chunk
            chunks_list.append(para)
            metadata_list.append(meta)
            sizes_list.append({"tokens": len(para_tokens), "chars": len(para)})
        else:
            # Paragraph too big → split by sentences
            sentences = sent_tokenize(para)
            current_tokens = []
            for sent in sentences:
                sent_tokens = enc.encode(sent)
                if len(current_tokens) + len(sent_tokens) > max_tokens:
                    # finalize current chunk
                    chunk_text = enc.decode(current_tokens)
                    chunks_list.append(chunk_text)
                    metadata_list.append(meta)
                    sizes_list.append({"tokens": len(current_tokens), "chars": len(chunk_text)})
                    # start new chunk
                    current_tokens = sent_tokens
                else:
                    current_tokens.extend(sent_tokens)

            # Add any leftover tokens
            if current_tokens:
                chunk_text = enc.decode(current_tokens)
                chunks_list.append(chunk_text)
                metadata_list.append(meta)
                sizes_list.append({"tokens": len(current_tokens), "chars": len(chunk_text)})

log_to_file(f"✅ Created {len(chunks_list)} chunks (sentence-safe for large paragraphs).")


# Filter out small chunks (<35 chars)
filtered_chunks_list = []
filtered_metadata_list = []
filtered_sizes_list = []

for chunk, meta, size in zip(chunks_list, metadata_list, sizes_list):
    if size["chars"] >= 35:
        filtered_chunks_list.append(chunk)
        filtered_metadata_list.append(meta)
        filtered_sizes_list.append(size)

# Replace old lists with filtered ones
chunks_list = filtered_chunks_list
metadata_list = filtered_metadata_list
sizes_list = filtered_sizes_list

log_to_file(f"✅ {len(chunks_list)} chunks remaining after dropping small ones (<35 chars).")








import pandas as pd

# Load CSV and clean column names
keywords_csv = "static_data/overall_keywords.csv"
df = pd.read_csv(keywords_csv)
df.columns = df.columns.str.strip()  # remove any extra spaces in headers

# Normalize disease name
disease_name_lookup = disease_name.replace("_", " ").lower()

extra_keywords = [
    # Original List
    "drug", "therapy", "symptom", "medication", "treat", "alleviate", "relief", "remedy", "prevalence",

    # Core Disease & Medical Condition Terms
    "disease", "illness", "sickness", "disorder", "syndrome", "condition", "ailment",

    # Types and Nature of Diseases
    "infectious", "viral", "bacterial", "fungal", "parasitic", "chronic", "acute", "terminal",
    "progressive", "degenerative", "autoimmune", "inflammatory", "deficiency", "genetic", "hereditary",
    "malignant", "benign", "metastatic", "outbreak", "epidemic", "pandemic",

    # Symptoms and Clinical Presentation
    "sign", "manifestation", "pain", "fever", "fatigue", "nausea", "vomiting", "diarrhea", "rash",
    "swelling", "inflammation", "diagnosis", "prognosis", "screening",

    # Causes and Origins
    "cause", "etiology", "pathogen", "virus", "bacteria", "fungus", "parasite", "congenital",
    "environmental", "lifestyle",

    # Treatment and Management
    "treatment", "pharmaceutical", "prescription", "cure", "vaccine", "vaccination", "immunization",
    "surgery", "operation", "procedure", "intervention", "dose", "dosage", "administration",
    "manage", "control", "ease", "mitigate", "prevent",

    # Measurement and Statistics
    "incidence", "morbidity", "mortality", "fatality rate", "risk factor", "comorbidity", "complication",

    # Body Systems and Anatomy
    "cardiovascular", "respiratory", "neurological", "gastrointestinal", "musculoskeletal", "endocrine", "renal", "hepatic"
]

# Find the row matching the disease name exactly
disease_row = df[df['Disease Name'].str.strip().str.lower() == disease_name_lookup]

if not disease_row.empty:
    # Get keywords from CSV
    csv_keywords = disease_row['Keywords'].values[0]
    csv_keywords_list = [kw.strip().lower() for kw in csv_keywords.split(',')]

    # Combine with extra keywords and deduplicate
    combined_keywords = list(set(csv_keywords_list + extra_keywords))
    print("Keywords:", combined_keywords)
else:
    print(f"No keywords found for disease: {disease_name_lookup}")


# Initialize new lists
filtered_chunks_with_keywords = []
filtered_metadata_with_keywords = []

# Lowercase keywords for case-insensitive matching
keywords_lower = [kw.lower() for kw in combined_keywords]

for chunk, meta in zip(filtered_chunks_list, filtered_metadata_list):
    chunk_lower = chunk.lower()  # lowercase the chunk
    # Check if any keyword is in the chunk
    if any(kw in chunk_lower for kw in keywords_lower):
        filtered_chunks_with_keywords.append(chunk)
        filtered_metadata_with_keywords.append(meta)

# Result: filtered_chunks_with_keywords and filtered_metadata_with_keywords are aligned
log_to_file("Number of chunks matched with keywords:", len(filtered_chunks_with_keywords), "out of", len(filtered_chunks_list), "from a set of", len(set(filtered_metadata_with_keywords)), "papers.")










import torch
import pandas as pd
import re, csv, os, sys
# from unsloth import FastLanguageModel
from transformers import TextStreamer
from vllm import LLM, SamplingParams
import warnings
from global_prompts import simple_sentence_prompt, global_output_max_new_tokens, triplets_from_simple_sentence_prompt

# Suppress the warning related to slow image processor
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_length = global_output_max_new_tokens
dtype = None
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
use_gpu = True

# Load model
if use_gpu:
    model = LLM(
        model='unsloth/gemma-3-27b-it-unsloth-bnb-4bit',
        max_model_len=90000,
        quantization="bitsandbytes",
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,

    )

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def extract_response(input_string):
    pattern = r'### Response:\n(.*?)(?=\n###|$)'  # Updated pattern to ensure it captures the full response
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        response = match.group(1).strip()  # Extract and clean the response
        return response



list_of_inputs = filtered_chunks_with_keywords #[0:5]
list_of_meta_data =  filtered_metadata_with_keywords #[0:5]
# list_of_inputs = list_of_inputs[0:3]


batch_size = 45  # adjust depending on GPU memory
responses_list = []
given_prompt = simple_sentence_prompt
def make_prompts(chunks):
    return [alpaca_prompt.format(given_prompt, chunk, "") for chunk in chunks]

# Process in batches
for i in range(0, len(list_of_inputs), batch_size):
    batch_inputs = list_of_inputs[i:i+batch_size]
    batch_prompts = make_prompts(batch_inputs)

    sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
    outputs = model.generate(batch_prompts, sampling_params)

    # Extract responses
    for output in outputs:
        full_text = output.prompt + output.outputs[0].text
        res = extract_response(full_text)
        responses_list.append(res)









# Remove the redundant simple sentences with the same meaning generated from the relevant text
from sentence_transformers import SentenceTransformer, util
# Load a sentence-transformer model optimized for paraphrase detection
model_to_find_unique_sentences = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def find_unique_sentences(list_of_sentences_here):
    # Get sentence embeddings
    embeddings_to_find_unique_sentences = model_to_find_unique_sentences.encode(list_of_sentences_here, convert_to_tensor=True)
    unique_sentences = []
    cosine_scores = util.pytorch_cos_sim(embeddings_to_find_unique_sentences, embeddings_to_find_unique_sentences)
    added = set()
    threshold = 0.95  # Similarity threshold

    for i in range(len(list_of_sentences_here)):
        if i in added:
            continue
        unique_sentences.append(list_of_sentences_here[i])
        for j in range(i + 1, len(list_of_sentences_here)):
            if cosine_scores[i][j] > threshold:
                added.add(j)
    return unique_sentences


number_of_cleaned_sentences = 0
number_of_unique_sentences = 0
new_response_list = []
all_unique_sentences = []
# New lists for mapping unique sentences to metadata
unique_sentences_list = []
unique_sentences_metadata_list = []

import re

def remove_gibberish(text):
    text = re.sub(r'["“”]+', '', text)
    text = re.sub(r'Progress in understanding.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PMC\d+', '', text)
    text = re.sub(r'\([^)]*\d[^)]*\)', '', text)
    text = text.strip(" ,.;")
    return text if len(text) >= 5 else None

# for res, meta in zip(responses_list, filtered_metadata_list):
for res, meta in zip(responses_list, list_of_meta_data):
    cleaned_sentences = []
    for line in res.splitlines():
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            line = line[1:-1].strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        sentence = remove_gibberish(line)
        if sentence:
            cleaned_sentences.append(sentence)

    number_of_cleaned_sentences += len(cleaned_sentences)
    unique_sentences = find_unique_sentences(cleaned_sentences)
    all_unique_sentences.append(unique_sentences)
    number_of_unique_sentences += len(unique_sentences)

    for us in unique_sentences:
        unique_sentences_list.append(us)
        unique_sentences_metadata_list.append(meta)

    formatted_string = "\n".join(f'("{sen}")' for sen in unique_sentences)
    new_response_list.append(formatted_string)


log_to_file("Number of simple sentences in LLM response:", number_of_cleaned_sentences)
log_to_file("Number of unique sentences found in final output:", number_of_unique_sentences)

output_file = 'results_after_year_2000/' + disease_name + '/generated_simple_sentences_' + disease_name + '.csv' 
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Relevant Text', 'Simple Sentences', 'MetaData'])
    for item1, item2, item3 in zip(list_of_inputs, new_response_list, list_of_meta_data):
        writer.writerow([item1, item2, item3])

print(f'CSV file "{output_file}" created successfully.')








input_for_triplet_generation = unique_sentences_list
metadata_for_triplet_generation = unique_sentences_metadata_list
print(f"Total unique simple sentences for triplet generation: {len(input_for_triplet_generation)}, {len(metadata_for_triplet_generation)}")

# Generate triplets for each simple sentence
given_prompt = triplets_from_simple_sentence_prompt
responses_list = [None for _ in input_for_triplet_generation]
prompts = []
for i in range(0,len(input_for_triplet_generation)):
    input_here = input_for_triplet_generation[i]

    prompts.append(alpaca_prompt.format(given_prompt, input_here, ""))

sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
outputs = model.generate(prompts, sampling_params)

# Process the outputs
responses_list = []
for i, output in enumerate(outputs):
    # The full text is needed for the regex to find the response block
    full_text = output.prompt + output.outputs[0].text
    res = extract_response(full_text)
    responses_list.append(res)
    # print(f"Processed prompt {i+1}/{len(prompts)}")




list_of_unformatted_simple_sentences =  input_for_triplet_generation
list_of_unformatted_triples = responses_list
list_of_unformatted_meta_data = metadata_for_triplet_generation

list_of_formatted_simple_sentences = []
list_of_formatted_triples = []
list_of_formatted_meta_data = []

for i in range(0,len(list_of_unformatted_triples)):
    lines = list_of_unformatted_triples[i].splitlines()
    cleaned_triples = []
    for line in lines:
        sentence = line.strip() # Remove leading and trailing whitespace
        if sentence.startswith('(') and sentence.endswith(')'): # Strip parentheses if they are present at the beginning and end
            sentence = sentence[1:-1].strip()
        if sentence.startswith('"') and sentence.endswith('"'): # Strip quotes if they are present at the beginning and end
            sentence = sentence[1:-1].strip()
        if sentence:  # Only add non-empty sentences
            cleaned_triples.append(sentence)
    if len(cleaned_triples) > 0:
        clean_length = len(cleaned_triples)     
        list_of_formatted_simple_sentences.extend([list_of_unformatted_simple_sentences[i]] * clean_length)
        list_of_formatted_triples.extend(cleaned_triples)
        list_of_formatted_meta_data.extend([list_of_unformatted_meta_data[i]] * clean_length)

print("Number of triples in the whole dataset that was generate by llm from simple sentences:")
log_to_file("Number of triples in the whole dataset that was generate by llm from simple sentences:", len(list_of_formatted_triples))
print(len(list_of_formatted_simple_sentences), len(list_of_formatted_triples), len(list_of_formatted_meta_data))


# # Create DataFrame
# df = pd.DataFrame({
#     "Simple_Sentence": list_of_formatted_simple_sentences,
#     "Triple": list_of_formatted_triples,
#     "Metadata": list_of_formatted_meta_data
# })

df = pd.DataFrame({
    "Simple_Sentence": list_of_formatted_simple_sentences,
    "Triple": list_of_formatted_triples
})
df["Metadata"] = [list_of_formatted_meta_data[i] if i < len(list_of_formatted_meta_data) else None
                  for i in range(len(df))]


# Save to CSV
df.to_csv('results_after_year_2000/' + disease_name + '/triples_' + disease_name + '.csv', index=False)



end_time = time.time()     # Record end time
elapsed = (end_time - start_time)/60  # Calculate elapsed time in minutes

log_to_file(f"The number of unique metadata items (papers used) is: {len(set(df['Metadata']))} for ", disease_name)
log_to_file(f"Time taken: {elapsed:.2f} minutes for ", disease_name)
log_to_file("~" * 100)
