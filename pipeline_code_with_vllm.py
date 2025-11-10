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
import json

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from vllm import LLM, SamplingParams

from global_prompts import (
    simple_sentence_prompt,
    global_output_max_new_tokens,
    triplets_from_simple_sentence_prompt
)
# Suppress warnings
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


# ============================================================================
# CONFIGURATION / CONSTANTS
# ============================================================================

def load_model_config(path):
    """Load model configuration from a JSON file.

    Returns a dict with configuration values. If the file does not exist or
    cannot be parsed the function returns an empty dict and the code using
    it will fall back to built-in defaults.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        print(f"Loaded model config from {path}")
        return cfg
    except Exception as e:
        print(f"Warning: could not load model config from {path}: {e}")
        return {}

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
def has_checkpoint_files(path):
    """Return True if the given path contains likely model checkpoint files."""
    if not os.path.isdir(path):
        return False
    filenames = os.listdir(path)
    # Common HF weight filenames or safe tensor names
    for name in filenames:
        if name.endswith('.bin') or name.endswith('.safetensors') or name.endswith('.gguf'):
            return True
    # Also accept directories that may contain shard files
    for name in filenames:
        if re.match(r'pytorch_model-\d{4}-of-\d{4}\.bin', name):
            return True
    return False

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


def find_unique_sentences(list_of_sentences_here, model_to_find_unique_sentences, similarity_threshold):
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
            if cosine_scores[i][j] > similarity_threshold:
                added.add(j)
    return unique_sentences


def make_prompts(chunks, given_prompt, alpaca_prompt):
    """Create alpaca formatted prompts from chunks."""
    return [alpaca_prompt.format(given_prompt, chunk, "") for chunk in chunks]


def load_keywords(disease_name):
    """Load disease-specific and general keywords."""
    # Base set of general medical and disease-related keywords used to filter chunks.
    # These are combined with any disease-specific keywords found in the CSV for a
    # stronger, wider matching set used by `filter_chunks_by_keywords`.
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


def get_hf_model_source(model_name, model_local_path):
    """determine model source: local path or Hugging Face name.

    Args:
        model_name (_type_): _description_
        model_local_path (_type_): _description_
    Returns:
        _type_: _description_
    """
    model_source = model_name
    if has_checkpoint_files(model_local_path):
        model_source = model_local_path
    else:
        print(f"No local model found at {model_local_path}, using  model name {model_name} from Hugging Face.")
        model_source = model_name
    return model_source
    
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution.

    The body below is divided into labeled steps for readability. The behavior
    is unchanged; comments only document the flow.
    """

    # Step 1: Start timer and parse command-line argument (disease name)
    start_time = time.time()
    disease_name = sys.argv[1]
    
    # Load config 
    # Path to optional JSON file containing model configuration overrides.
    MODEL_CONFIG_PATH = os.path.join(os.getcwd(), "model_config.json")
    model_config = load_model_config(MODEL_CONFIG_PATH)

    # GEMMA model configuration (defaults may be overridden by models/model_config.json)
    llm_model_name = model_config['llm_model_name']
    llm_model_path = model_config['llm_model_path']
    llm_max_model_len = model_config['llm_max_model_len']
    quantization = model_config["quantization"]
    gpu_memory_utilization = model_config["gpu_memory_utilization"]
    tensor_parallel_size = model_config["tensor_parallel_size"]

    # Processing configuration
    batch_size = model_config["batch_size"]
    similarity_threshold = model_config["similarity_threshold"]

    # sentence transformer model name
    sentence_transformer_model = model_config["sentence_transformer_model"]
    sentence_transformer_model_path = model_config["sentence_transformer_model_path"]
    # Local path where the sentence-transformers repo is cloned by the download script

    # File paths
    LOG_FILE = "DRAC_output_log_from_pipeline.log"
    KEYWORDS_CSV = "static_data/overall_keywords.csv"

    log_to_file("Processing disease: " + disease_name)

    # Step 2: Load precomputed chunks and their metadata from CSV
    csv_filename = f"data/{disease_name}/{disease_name}_input_chunks_and_metadata.csv"
    df_loaded = pd.read_csv(csv_filename)
    filtered_chunks_list = df_loaded["Filtered_Chunks"].tolist()
    filtered_metadata_list = df_loaded["Filtered_Metadata"].tolist()

    # Step 3: Load keywords (general + disease-specific) and filter chunks
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

    # Step 4: Load the LLM model (GPU required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU is required for this pipeline.")

    # Prefer a local gemma clone if it exists (downloaded into models/huggingface/...)
    gemma_model_source = get_hf_model_source(llm_model_name, llm_model_path)
    # Initialize the vLLM models
    llm_model = LLM(
        model=gemma_model_source,
        max_model_len=llm_max_model_len,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Step 5: Prepare prompts and generate simple sentences with the LLM (batched)
    list_of_inputs = filtered_chunks_with_keywords
    list_of_meta_data = filtered_metadata_with_keywords
    responses_list = []
    given_prompt = simple_sentence_prompt

    for i in range(0, len(list_of_inputs), batch_size):
        batch_inputs = list_of_inputs[i:i + batch_size]
        batch_prompts = make_prompts(batch_inputs, given_prompt, ALPACA_PROMPT_TEMPLATE)

        sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
        outputs = llm_model.generate(batch_prompts, sampling_params)

        # Extract responses from model outputs
        for output in outputs:
            full_text = output.prompt + output.outputs[0].text
            res = extract_response(full_text)
            responses_list.append(res)

    # Step 6: Deduplicate simple sentences using a sentence-transformer
    sentnce_model_source = get_hf_model_source(sentence_transformer_model, sentence_transformer_model_path)
    model_to_find_unique_sentences = SentenceTransformer(sentnce_model_source)

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
        unique_sentences = find_unique_sentences(cleaned_sentences, model_to_find_unique_sentences, similarity_threshold)
        number_of_unique_sentences += len(unique_sentences)

        for us in unique_sentences:
            unique_sentences_list.append(us)
            unique_sentences_metadata_list.append(meta)

        formatted_string = "\n".join(f'("{sen}")' for sen in unique_sentences)
        new_response_list.append(formatted_string)

    log_to_file("Number of simple sentences in LLM response:", number_of_cleaned_sentences)
    log_to_file("Number of unique sentences found in final output:", number_of_unique_sentences)

    # Step 7: Save the generated simple sentences to CSV
    output_file = f'data/{disease_name}/generated_simple_sentences_{disease_name}.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Relevant Text', 'Simple Sentences', 'MetaData'])
        for item1, item2, item3 in zip(list_of_inputs, new_response_list, list_of_meta_data):
            writer.writerow([item1, item2, item3])

    print(f'CSV file "{output_file}" created successfully.')

    # Step 8: Generate triplets from unique simple sentences
    input_for_triplet_generation = unique_sentences_list
    metadata_for_triplet_generation = unique_sentences_metadata_list
    print(f"Total unique simple sentences for triplet generation: {len(input_for_triplet_generation)}, {len(metadata_for_triplet_generation)}")

    given_prompt = triplets_from_simple_sentence_prompt
    prompts = []
    for i in range(0, len(input_for_triplet_generation)):
        input_here = input_for_triplet_generation[i]
        prompts.append(ALPACA_PROMPT_TEMPLATE.format(given_prompt, input_here, ""))

    sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
    outputs = llm_model.generate(prompts, sampling_params)

    # Extract triplet responses
    responses_list = []
    for i, output in enumerate(outputs):
        full_text = output.prompt + output.outputs[0].text
        res = extract_response(full_text)
        responses_list.append(res)

    # Step 9: Format triplets and create DataFrame
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

    df = pd.DataFrame({
        "Simple_Sentence": list_of_formatted_simple_sentences,
        "Triple": list_of_formatted_triples
    })
    df["Metadata"] = [
        list_of_formatted_meta_data[i] if i < len(list_of_formatted_meta_data) else None
        for i in range(len(df))
    ]

    # Step 10: Save triples and log final statistics
    df.to_csv(f'data/{disease_name}/triples_{disease_name}.csv', index=False)

    end_time = time.time()
    elapsed = (end_time - start_time) / 60

    log_to_file(f"The number of unique metadata items (papers used) is: {len(set(df['Metadata']))} for ", disease_name)
    log_to_file(f"Time taken: {elapsed:.2f} minutes for ", disease_name)
    log_to_file("~" * 100)


if __name__ == "__main__":
    main()
