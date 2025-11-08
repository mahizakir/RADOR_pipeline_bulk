"""
RADOR Pipeline - Medical Text Processing with vLLM
"""

# ============================================================================
# ALL IMPORTS
# ============================================================================
import os
import sys
import csv
import re
import time
import warnings

import pandas as pd
import torch
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import TextStreamer
from vllm import LLM, SamplingParams

from global_prompts import (
    simple_sentence_prompt,
    global_output_max_new_tokens,
    triplets_from_simple_sentence_prompt
)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


# ============================================================================
# CONFIGURATION / CONSTANTS
# ============================================================================

# Model configuration
LLM_MODEL_NAME = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
MAX_MODEL_LEN = 90000
QUANTIZATION = "bitsandbytes"
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 1

# Processing configuration
BATCH_SIZE = 45
SIMILARITY_THRESHOLD = 0.95

# Model names
SENTENCE_TRANSFORMER_MODEL = "paraphrase-MiniLM-L6-v2"

# File paths
LOG_FILE = "DRAC_output_log_from_pipeline.log"
KEYWORDS_CSV = "static_data/overall_keywords.csv"

# Alpaca prompt template
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# ============================================================================
# ALL FUNCTIONS
# ============================================================================

def log_to_file(*args, file_path=LOG_FILE):
    """Function to print and append only selected messages."""
    with open(file_path, "a") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(*args)
        sys.stdout = old_stdout


def extract_response(input_string):
    """Extract response from alpaca formatted output."""
    pattern = r'### Response:\n(.*?)(?=\n###|$)'
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        response = match.group(1).strip()
        return response
    return None


def remove_gibberish(text):
    """Remove unwanted patterns from text."""
    text = re.sub(r'["""]+', '', text)
    text = re.sub(r'Progress in understanding.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PMC\d+', '', text)
    text = re.sub(r'\([^)]*\d[^)]*\)', '', text)
    text = text.strip(" ,.;")
    return text if len(text) >= 5 else None


def find_unique_sentences(list_of_sentences_here, model_to_find_unique_sentences):
    """Find unique sentences using cosine similarity."""
    embeddings_to_find_unique_sentences = model_to_find_unique_sentences.encode(
        list_of_sentences_here, convert_to_tensor=True
    )
    unique_sentences = []
    cosine_scores = util.pytorch_cos_sim(
        embeddings_to_find_unique_sentences, 
        embeddings_to_find_unique_sentences
    )
    added = set()

    for i in range(len(list_of_sentences_here)):
        if i in added:
            continue
        unique_sentences.append(list_of_sentences_here[i])
        for j in range(i + 1, len(list_of_sentences_here)):
            if cosine_scores[i][j] > SIMILARITY_THRESHOLD:
                added.add(j)
    return unique_sentences


def make_prompts(chunks, given_prompt, alpaca_prompt):
    """Create alpaca formatted prompts from chunks."""
    return [alpaca_prompt.format(given_prompt, chunk, "") for chunk in chunks]


def load_keywords(disease_name):
    """Load disease-specific and general keywords."""
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
        "cardiovascular", "respiratory", "neurological", "gastrointestinal", "musculoskeletal", 
        "endocrine", "renal", "hepatic"
    ]
    
    df = pd.read_csv(KEYWORDS_CSV)
    df.columns = df.columns.str.strip()
    
    disease_name_lookup = disease_name.replace("_", " ").lower()
    disease_row = df[df['Disease Name'].str.strip().str.lower() == disease_name_lookup]
    combined_keywords = extra_keywords.copy()
    
    if not disease_row.empty:
        csv_keywords = disease_row['Keywords'].values[0]
        csv_keywords_list = [kw.strip().lower() for kw in csv_keywords.split(',')]
        combined_keywords = list(set(csv_keywords_list + extra_keywords))
        print("Keywords:", combined_keywords)
    else:
        print(f"No keywords found for disease: {disease_name_lookup}")
    
    return combined_keywords


def filter_chunks_by_keywords(chunks_list, metadata_list, keywords):
    """Filter chunks that contain any of the keywords."""
    filtered_chunks = []
    filtered_metadata = []
    keywords_lower = [kw.lower() for kw in keywords]
    
    for chunk, meta in zip(chunks_list, metadata_list):
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in keywords_lower):
            filtered_chunks.append(chunk)
            filtered_metadata.append(meta)
    
    return filtered_chunks, filtered_metadata


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution."""
    start_time = time.time()
    
    # Get disease name from command line
    disease_name = sys.argv[1]
    log_to_file("Processing disease: " + disease_name)
    
    # Load chunks and metadata
    csv_filename = f"data/{disease_name}/{disease_name}_input_chunks_and_metadata.csv"
    df_loaded = pd.read_csv(csv_filename)
    filtered_chunks_list = df_loaded["Filtered_Chunks"].tolist()
    filtered_metadata_list = df_loaded["Filtered_Metadata"].tolist()
    
    # Load keywords and filter chunks
    combined_keywords = load_keywords(disease_name)
    filtered_chunks_with_keywords, filtered_metadata_with_keywords = filter_chunks_by_keywords(
        filtered_chunks_list, filtered_metadata_list, combined_keywords
    )
    
    log_to_file(
        "Number of chunks matched with keywords:", 
        len(filtered_chunks_with_keywords), 
        "out of", 
        len(filtered_chunks_list), 
        "from a set of", 
        len(set(filtered_metadata_with_keywords)), 
        "papers."
    )
    
    # Load model
    if torch.cuda.is_available():
        model = LLM(
            model=LLM_MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            quantization=QUANTIZATION,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        )
    else:
        raise RuntimeError("CUDA not available. GPU is required for this pipeline.")
    
    # Generate simple sentences
    list_of_inputs = filtered_chunks_with_keywords
    list_of_meta_data = filtered_metadata_with_keywords
    
    responses_list = []
    given_prompt = simple_sentence_prompt
    
    # Process in batches
    for i in range(0, len(list_of_inputs), BATCH_SIZE):
        batch_inputs = list_of_inputs[i:i+BATCH_SIZE]
        batch_prompts = make_prompts(batch_inputs, given_prompt, ALPACA_PROMPT_TEMPLATE)
        
        sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
        outputs = model.generate(batch_prompts, sampling_params)
        
        # Extract responses
        for output in outputs:
            full_text = output.prompt + output.outputs[0].text
            res = extract_response(full_text)
            responses_list.append(res)
    
    # Load sentence transformer model for deduplication
    model_to_find_unique_sentences = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    
    # Process and deduplicate simple sentences
    number_of_cleaned_sentences = 0
    number_of_unique_sentences = 0
    new_response_list = []
    unique_sentences_list = []
    unique_sentences_metadata_list = []
    
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
        unique_sentences = find_unique_sentences(cleaned_sentences, model_to_find_unique_sentences)
        number_of_unique_sentences += len(unique_sentences)
        
        for us in unique_sentences:
            unique_sentences_list.append(us)
            unique_sentences_metadata_list.append(meta)
        
        formatted_string = "\n".join(f'("{sen}")' for sen in unique_sentences)
        new_response_list.append(formatted_string)
    
    log_to_file("Number of simple sentences in LLM response:", number_of_cleaned_sentences)
    log_to_file("Number of unique sentences found in final output:", number_of_unique_sentences)
    
    # Save simple sentences to CSV
    output_file = f'data/{disease_name}/generated_simple_sentences_{disease_name}.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Relevant Text', 'Simple Sentences', 'MetaData'])
        for item1, item2, item3 in zip(list_of_inputs, new_response_list, list_of_meta_data):
            writer.writerow([item1, item2, item3])
    
    print(f'CSV file "{output_file}" created successfully.')
    
    # Generate triplets
    input_for_triplet_generation = unique_sentences_list
    metadata_for_triplet_generation = unique_sentences_metadata_list
    print(f"Total unique simple sentences for triplet generation: {len(input_for_triplet_generation)}, {len(metadata_for_triplet_generation)}")
    
    given_prompt = triplets_from_simple_sentence_prompt
    responses_list = [None for _ in input_for_triplet_generation]
    prompts = []
    
    for i in range(0, len(input_for_triplet_generation)):
        input_here = input_for_triplet_generation[i]
        prompts.append(ALPACA_PROMPT_TEMPLATE.format(given_prompt, input_here, ""))
    
    sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
    outputs = model.generate(prompts, sampling_params)
    
    # Process the outputs
    responses_list = []
    for i, output in enumerate(outputs):
        full_text = output.prompt + output.outputs[0].text
        res = extract_response(full_text)
        responses_list.append(res)
    
    # Format triplets
    list_of_unformatted_simple_sentences = input_for_triplet_generation
    list_of_unformatted_triples = responses_list
    list_of_unformatted_meta_data = metadata_for_triplet_generation
    
    list_of_formatted_simple_sentences = []
    list_of_formatted_triples = []
    list_of_formatted_meta_data = []
    
    for i in range(0, len(list_of_unformatted_triples)):
        lines = list_of_unformatted_triples[i].splitlines()
        cleaned_triples = []
        for line in lines:
            sentence = line.strip()
            if sentence.startswith('(') and sentence.endswith(')'):
                sentence = sentence[1:-1].strip()
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1].strip()
            if sentence:
                cleaned_triples.append(sentence)
        if len(cleaned_triples) > 0:
            clean_length = len(cleaned_triples)
            list_of_formatted_simple_sentences.extend([list_of_unformatted_simple_sentences[i]] * clean_length)
            list_of_formatted_triples.extend(cleaned_triples)
            list_of_formatted_meta_data.extend([list_of_unformatted_meta_data[i]] * clean_length)
    
    print("Number of triples in the whole dataset that was generate by llm from simple sentences:")
    log_to_file("Number of triples in the whole dataset that was generate by llm from simple sentences:", len(list_of_formatted_triples))
    print(len(list_of_formatted_simple_sentences), len(list_of_formatted_triples), len(list_of_formatted_meta_data))
    
    # Create DataFrame
    df = pd.DataFrame({
        "Simple_Sentence": list_of_formatted_simple_sentences,
        "Triple": list_of_formatted_triples
    })
    df["Metadata"] = [
        list_of_formatted_meta_data[i] if i < len(list_of_formatted_meta_data) else None
        for i in range(len(df))
    ]
    
    # Save to CSV
    df.to_csv(f'data/{disease_name}/triples_{disease_name}.csv', index=False)
    
    # Final statistics
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    
    log_to_file(f"The number of unique metadata items (papers used) is: {len(set(df['Metadata']))} for ", disease_name)
    log_to_file(f"Time taken: {elapsed:.2f} minutes for ", disease_name)
    log_to_file("~" * 100)


if __name__ == "__main__":
    main()
