"""
RADOR Pipeline - Medical Text Processing with vLLM
"""
# ============================================================================
# ALL IMPORTS
# ============================================================================
import os
import argparse
import csv
import re
import time
import warnings
import json
import logging

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
        # defer to module logger if available
        logging.getLogger(__name__).info("Loaded model config from %s", path)
        return cfg
    except Exception as e:
        logging.getLogger(__name__).warning("Could not load model config from %s: %s", path, e)
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

def init_logger(log_file, level=logging.INFO):
    """Initialize and return a logger for the pipeline.

    Writes logs to stdout and to `log_file`. Safe to call multiple times.
    """
    logger = logging.getLogger("rador")
    logger.setLevel(level)
    # remove existing handlers to avoid duplicate logs
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False

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


def load_keywords(keyword_csv, disease_name):
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
    logger = logging.getLogger("rador")
    df = pd.read_csv(keyword_csv)
    df.columns = df.columns.str.strip()
    
    disease_name_lookup = disease_name.replace("_", " ").lower()
    disease_row = df[df['Disease Name'].str.strip().str.lower() == disease_name_lookup]
    combined_keywords = extra_keywords.copy()
    
    if not disease_row.empty:
        csv_keywords = disease_row['Keywords'].values[0]
        csv_keywords_list = [kw.strip().lower() for kw in csv_keywords.split(',')]
        combined_keywords = list(set(csv_keywords_list + extra_keywords))
        logger.info("Keywords: %s", combined_keywords)
    else:
        logger.warning("No keywords found for disease: %s", disease_name_lookup)
    
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
    logger = logging.getLogger("rador")
    model_source = model_name
    if has_checkpoint_files(model_local_path):
        model_source = model_local_path
    else:
        logger.info("No local model found at %s, using model name %s from Hugging Face.", model_local_path, model_name)
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

    # Step 1: Start timer and parse command-line arguments
    parser = argparse.ArgumentParser(description="RADOR pipeline: generate triples from disease documents")
    parser.add_argument(
        "-d", "--disease_name",
        dest="disease_name",
        required=True,
        help="Name of the disease to process (e.g., alzheimers_disease)"
    )
    parser.add_argument(
        "-i", "--input_csv",
        dest="input_csv",
        default=None,
        help="Path to input CSV containing precomputed chunks and metadata. Defaults to data/<disease_name>/<disease_name>_input_chunks_and_metadata.csv"
    )
    parser.add_argument(
        "-o", "--triples_out",
        dest="triples_out",
        default=None,
        help="Path to save triples CSV. Defaults to data/<disease_name>/triples_<disease_name>.csv"
    )

    args = parser.parse_args()

    start_time = time.time()
    disease_name = args.disease_name
    triples_out_path = args.triples_out or f"data/{disease_name}/triples_{disease_name}.csv"
    
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

    log_file_fp = model_config["log_file"]
    keywords_csv_fp = model_config["keywords_csv"]
    
    # Initialize logger
    init_logger(log_file_fp, level=logging.INFO)
    # Disable vLLM compile cache to avoid permission/compatibility issues in some environments.
    # This is safe: vLLM will skip using the compiled cache and fall back to runtime compilation.
    # if os.environ.get("VLLM_DISABLE_COMPILE_CACHE") != "1":
    #    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    #    logging.getLogger("rador").info("Set VLLM_DISABLE_COMPILE_CACHE=1 to disable vLLM compile cache")
    
    # File paths
    logger = logging.getLogger("rador")
    logger.info("Processing disease: %s", disease_name)
    # exit()
    # Step 2: Load precomputed chunks and their metadata from CSV
    csv_filename = args.input_csv or f"data/{disease_name}/{disease_name}_input_chunks_and_metadata.csv"
    logger.info("Reading input CSV from %s", csv_filename)
    if not os.path.isfile(csv_filename):
        logger.error("Input CSV not found: %s", csv_filename)
        raise FileNotFoundError(f"Input CSV not found: {csv_filename}")
    df_loaded = pd.read_csv(csv_filename)
    filtered_chunks_list = df_loaded["Filtered_Chunks"].tolist()
    filtered_metadata_list = df_loaded["Filtered_Metadata"].tolist()

    # Step 3: Load keywords (general + disease-specific) and filter chunks
    combined_keywords = load_keywords(keywords_csv_fp, disease_name)
    filtered_chunks_with_keywords, filtered_metadata_with_keywords = filter_chunks_by_keywords(
        filtered_chunks_list, filtered_metadata_list, combined_keywords
    )

    logger.info(
        "Number of chunks matched with keywords: %d out of %d from a set of %d papers.",
        len(filtered_chunks_with_keywords),
        len(filtered_chunks_list),
        len(set(filtered_metadata_with_keywords)),
    )

    # Step 4: Load the LLM model (GPU required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU is required for this pipeline.")

    # Prefer a local gemma clone if it exists (downloaded into models/huggingface/...)
    gemma_model_source = get_hf_model_source(llm_model_name, llm_model_path)

    # Initialize the vLLM model
    # Construct vLLM LLM with explicit generate runner. Some model repos expose only
    # If the model truly does not support generation (e.g., it's a sentence-transformer),
    # vLLM will raise a ValueError when `.generate()` is called and we handle it below.
    logger.info("Loading LLM model from source: %s", gemma_model_source)
    llm_model = LLM(
        model=gemma_model_source,
        #runner="generate", 
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
        try:
            outputs = llm_model.generate(batch_prompts, sampling_params)
        except ValueError as e:
            logger.error("vLLM generate() failed: %s", e)
            logger.error(
                "This model may not support generation (it could be an embedding-only repo or missing weights). "
                "Check that `%s` points to a generative model and that git-lfs objects are present."
                , gemma_model_source
            )
            raise

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

    logger.info("Number of simple sentences in LLM response: %d", number_of_cleaned_sentences)
    logger.info("Number of unique sentences found in final output: %d", number_of_unique_sentences)

    # Step 7: Save the generated simple sentences to CSV
    output_file = f'data/{disease_name}/generated_simple_sentences_{disease_name}.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Relevant Text', 'Simple Sentences', 'MetaData'])
        for item1, item2, item3 in zip(list_of_inputs, new_response_list, list_of_meta_data):
            writer.writerow([item1, item2, item3])

    logger.info('CSV file "%s" created successfully.', output_file)

    # Step 8: Generate triplets from unique simple sentences
    input_for_triplet_generation = unique_sentences_list
    metadata_for_triplet_generation = unique_sentences_metadata_list
    logger.info("Total unique simple sentences for triplet generation: %d, %d", len(input_for_triplet_generation), len(metadata_for_triplet_generation))

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

    logger.info("Number of triples in the whole dataset that was generated by llm from simple sentences: %d", len(list_of_formatted_triples))
    logger.info("Triples / simple_sentences / metadata counts: %d %d %d", len(list_of_formatted_simple_sentences), len(list_of_formatted_triples), len(list_of_formatted_meta_data))

    df = pd.DataFrame({
        "Simple_Sentence": list_of_formatted_simple_sentences,
        "Triple": list_of_formatted_triples
    })
    df["Metadata"] = [
        list_of_formatted_meta_data[i] if i < len(list_of_formatted_meta_data) else None
        for i in range(len(df))
    ]

    # Step 10: Save triples and log final statistics
    # Ensure output directory exists
    out_dir = os.path.dirname(triples_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    logger.info("Saving triples to %s", triples_out_path)
    df.to_csv(triples_out_path, index=False)

    end_time = time.time()
    elapsed = (end_time - start_time) / 60

    logger.info("The number of unique metadata items (papers used) is: %d for %s", len(set(df['Metadata'])), disease_name)
    logger.info("Time taken: %.2f minutes for %s", elapsed, disease_name)
    logger.info("%s", "~" * 100)

if __name__ == "__main__":

    logger = logging.getLogger("rador")
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA version seen by torch: %s", torch.version.cuda)
    if torch.cuda.is_available():
        try:
            logger.info("Device: %s", torch.cuda.get_device_name(0))
            logger.info("Capability: %s", torch.cuda.get_device_capability(0))
        except Exception:
            logger.exception("Failed to query CUDA device properties")
    else:
        logger.info("No CUDA device available")
    logger.info("CUDA_LAUNCH_BLOCKING = %s", os.environ.get("CUDA_LAUNCH_BLOCKING"))

    main()
