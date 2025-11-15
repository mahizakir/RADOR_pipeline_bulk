#!/usr/bin/env python3
"""
pipeline_transformers_adapter.py

An adapter pipeline that mirrors the vLLM-based pipeline but uses
Hugging Face `transformers` for generation. This is provided as a
separate file so you can switch without modifying the original vLLM
implementation.

Usage:
    python pipeline_transformers_adapter.py <disease_name>

Notes:
- By default the adapter will try to use 4-bit quantization via
  `bitsandbytes` if available. If bitsandbytes is not installed the
  script will fall back to a full-precision load (may require lots of GPU
  memory).
- The script expects the same CSV layout as `pipeline_code_with_vllm.py`:
  `data/<disease>/..._input_chunks_and_metadata.csv` with columns
  `Filtered_Chunks` and `Filtered_Metadata`.
"""
import sys
import csv
import re
import time
from typing import List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import GenerationConfig
except Exception:
    raise RuntimeError("Please install transformers (pip install 'transformers[torch]') to use this adapter")

try:
    from bitsandbytes import __version__ as bnb_version  # type: ignore
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# Configuration
HF_MODEL_NAME = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
SENTENCE_TRANSFORMER_MODEL = "paraphrase-MiniLM-L6-v2"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 0.95
BATCH_SIZE = 8
SIMILARITY_THRESHOLD = 0.95

# File paths
LOG_FILE = "DRAC_output_log_from_pipeline.log"
KEYWORDS_CSV = "static_data/overall_keywords.csv"

ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def log_to_file(*args, file_path=LOG_FILE):
    with open(file_path, "a") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(*args)
        sys.stdout = old_stdout


def extract_response(input_string: str) -> str:
    pattern = r"### Response:\n(.*?)(?=\n###|$)"
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return input_string


def remove_gibberish(text: str) -> str:
    text = re.sub(r'["“”]+', '', text)
    text = re.sub(r'Progress in understanding.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'PMC\d+', '', text)
    text = re.sub(r'\([^)]*\d[^)]*\)', '', text)
    text = text.strip(' ,.;')
    return text if len(text) >= 5 else None


def find_unique_sentences(list_of_sentences_here: List[str], model_to_find_unique_sentences):
    if not list_of_sentences_here:
        return []
    embeddings = model_to_find_unique_sentences.encode(list_of_sentences_here, convert_to_tensor=True)
    unique_sentences = []
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    added = set()
    for i in range(len(list_of_sentences_here)):
        if i in added:
            continue
        unique_sentences.append(list_of_sentences_here[i])
        for j in range(i + 1, len(list_of_sentences_here)):
            if cosine_scores[i][j] > SIMILARITY_THRESHOLD:
                added.add(j)
    return unique_sentences


def make_prompts(chunks: List[str], given_prompt: str) -> List[str]:
    return [ALPACA_PROMPT_TEMPLATE.format(given_prompt, chunk, "") for chunk in chunks]


def init_generation_model(model_name: str, load_in_4bit: bool = True):
    """Initialize tokenizer and model for generation.

    Returns (tokenizer, model, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_in_4bit and BNB_AVAILABLE:
        # Use bitsandbytes 4-bit loading via transformers if available.
        # This requires a compatible bitsandbytes + CUDA setup in the environment.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
            return tokenizer, model, device
        except Exception as e:
            print("Warning: failed to load in 4-bit mode, falling back to full precision:", e)

    # Fallback: full precision load (may require large GPU RAM)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    return tokenizer, model, device


def generate_with_transformers(tokenizer, model, prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS,
                               temperature: float = TEMPERATURE, top_p: float = TOP_P):
    outputs = []
    model.eval()
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
        outputs.extend(texts)
    return outputs


def load_keywords(disease_name: str):
    extra_keywords = [
        "drug", "therapy", "symptom", "medication", "treat", "alleviate", "relief", "remedy", "prevalence",
        "disease", "illness", "sickness", "disorder", "syndrome", "condition", "ailment",
        "infectious", "viral", "bacterial", "fungal", "parasitic", "chronic", "acute", "terminal",
        "progressive", "degenerative", "autoimmune", "inflammatory", "deficiency", "genetic", "hereditary",
        "malignant", "benign", "metastatic", "outbreak", "epidemic", "pandemic",
        "sign", "manifestation", "pain", "fever", "fatigue", "nausea", "vomiting", "diarrhea", "rash",
        "swelling", "inflammation", "diagnosis", "prognosis", "screening",
        "cause", "etiology", "pathogen", "virus", "bacteria", "fungus", "parasite", "congenital",
        "environmental", "lifestyle",
        "treatment", "pharmaceutical", "prescription", "cure", "vaccine", "vaccination", "immunization",
        "surgery", "operation", "procedure", "intervention", "dose", "dosage", "administration",
        "manage", "control", "ease", "mitigate", "prevent",
        "incidence", "morbidity", "mortality", "fatality rate", "risk factor", "comorbidity", "complication",
        "cardiovascular", "respiratory", "neurological", "gastrointestinal", "musculoskeletal",
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
    return combined_keywords


def filter_chunks_by_keywords(chunks_list: List[str], metadata_list: List[str], keywords: List[str]):
    filtered_chunks = []
    filtered_metadata = []
    keywords_lower = [kw.lower() for kw in keywords]
    for chunk, meta in zip(chunks_list, metadata_list):
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in keywords_lower):
            filtered_chunks.append(chunk)
            filtered_metadata.append(meta)
    return filtered_chunks, filtered_metadata


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_transformers_adapter.py <disease_name>")
        sys.exit(1)
    disease_name = sys.argv[1]
    start_time = time.time()
    log_to_file("Processing disease: " + disease_name)

    csv_filename = f"data/{disease_name}/{disease_name}_input_chunks_and_metadata.csv"
    df_loaded = pd.read_csv(csv_filename)
    filtered_chunks_list = df_loaded["Filtered_Chunks"].tolist()
    filtered_metadata_list = df_loaded["Filtered_Metadata"].tolist()

    combined_keywords = load_keywords(disease_name)
    filtered_chunks_with_keywords, filtered_metadata_with_keywords = filter_chunks_by_keywords(
        filtered_chunks_list, filtered_metadata_list, combined_keywords
    )

    log_to_file(
        "Number of chunks matched with keywords:",
        len(filtered_chunks_with_keywords),
        "out of",
        len(filtered_chunks_list),
    )

    # Initialize generation model
    tokenizer, model, device = init_generation_model(HF_MODEL_NAME, load_in_4bit=BNB_AVAILABLE)

    # Prepare prompts and generate
    given_prompt = "Simplify the following text into short, informative sentences."
    prompts = make_prompts(filtered_chunks_with_keywords, given_prompt)
    generated_texts = generate_with_transformers(tokenizer, model, prompts)

    # Extract responses and deduplicate
    responses_list = [extract_response(t) for t in generated_texts]

    sent_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    number_of_cleaned_sentences = 0
    number_of_unique_sentences = 0
    new_response_list = []
    unique_sentences_list = []
    unique_sentences_metadata_list = []

    for res, meta in zip(responses_list, filtered_metadata_with_keywords):
        cleaned_sentences = []
        for line in (res or "").splitlines():
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1].strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].strip()
            sentence = remove_gibberish(line)
            if sentence:
                cleaned_sentences.append(sentence)

        number_of_cleaned_sentences += len(cleaned_sentences)
        unique_sentences = find_unique_sentences(cleaned_sentences, sent_model)
        number_of_unique_sentences += len(unique_sentences)
        for us in unique_sentences:
            unique_sentences_list.append(us)
            unique_sentences_metadata_list.append(meta)
        formatted_string = "\n".join(f'(\"{sen}\")' for sen in unique_sentences)
        new_response_list.append(formatted_string)

    log_to_file("Number of simple sentences in LLM response:", number_of_cleaned_sentences)
    log_to_file("Number of unique sentences found in final output:", number_of_unique_sentences)

    # Save simple sentences
    output_file = f'data/{disease_name}/generated_simple_sentences_{disease_name}.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Relevant Text', 'Simple Sentences', 'MetaData'])
        for item1, item2, item3 in zip(filtered_chunks_with_keywords, new_response_list, filtered_metadata_with_keywords):
            writer.writerow([item1, item2, item3])

    # Generate triplets from unique sentences
    given_triplet_prompt = "Extract structured triples (subject, predicate, object) from the sentence."
    triplet_prompts = [ALPACA_PROMPT_TEMPLATE.format(given_triplet_prompt, s, "") for s in unique_sentences_list]
    triplet_texts = generate_with_transformers(tokenizer, model, triplet_prompts)
    triplet_responses = [extract_response(t) for t in triplet_texts]

    list_of_formatted_simple_sentences = []
    list_of_formatted_triples = []
    list_of_formatted_meta_data = []

    for i in range(len(triplet_responses)):
        lines = (triplet_responses[i] or "").splitlines()
        cleaned_triples = []
        for line in lines:
            sentence = line.strip()
            if sentence.startswith('(') and sentence.endswith(')'):
                sentence = sentence[1:-1].strip()
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1].strip()
            if sentence:
                cleaned_triples.append(sentence)
        if cleaned_triples:
            clean_length = len(cleaned_triples)
            list_of_formatted_simple_sentences.extend([unique_sentences_list[i]] * clean_length)
            list_of_formatted_triples.extend(cleaned_triples)
            list_of_formatted_meta_data.extend([unique_sentences_metadata_list[i]] * clean_length)

    df = pd.DataFrame({
        "Simple_Sentence": list_of_formatted_simple_sentences,
        "Triple": list_of_formatted_triples
    })
    df["Metadata"] = [list_of_formatted_meta_data[i] if i < len(list_of_formatted_meta_data) else None for i in range(len(df))]
    df.to_csv(f'data/{disease_name}/triples_{disease_name}.csv', index=False)

    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    log_to_file(f"Time taken: {elapsed:.2f} minutes for ", disease_name)
    log_to_file("~" * 100)


if __name__ == '__main__':
    main()
