
import pandas as pd
import torch
import re, csv, os, sys
from vllm import SamplingParams
from sentence_transformers import util
import warnings
from global_prompts import global_output_max_new_tokens, tagging_triples_prompt_with_relevant_sentences
from utils.model_utils import load_gemma_from_config, load_model_sentence_transformer, load_model_config
import logging


# Logger initialization (similar to paper_processor)
def init_logger(log_file, level=logging.INFO):
	"""Initialize and return a logger for the tagger."""
	logger = logging.getLogger("tagger")
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
	return logger

def categorize_relation(rel):
	rel_embed = model_sentence_transformers.encode(rel)
	best_cat, best_score = None, float("-inf")
	for cat, cat_embeds in category_embeddings.items():
		score = util.cos_sim(rel_embed, cat_embeds).max().item()
		if score > best_score:
			best_cat, best_score = cat, score
	return best_cat

def read_disease_names(disease_csv, start=0, limit=None):
	df = pd.read_csv(disease_csv, header=None)
	names = df[0].tolist()
	names = names[start:]       # skip first 'start' diseases
	if limit:
		names = names[:limit]   # take at most 'limit' diseases
	return names


def process_triples_file(triples_file):
	"""Read and clean a triples CSV, parse Entity1, Relation, Entity2."""
	if not os.path.exists(triples_file):
		return None, 0, 0, 0, 0  # file missing, return empty stats

	df = pd.read_csv(triples_file)
	total_before_drop = len(df)

	# Drop NaN triples
	nan_count = df['Triple'].isna().sum()
	df = df.dropna(subset=['Triple']).reset_index(drop=True)

	# Identify malformed triples
	drop_indices = []
	for idx, triple in enumerate(df['Triple']):
		parts = [p.strip().strip('"') for p in str(triple).split(',')]
		if len(parts) != 3:
			drop_indices.append(idx)

	malformed_count = len(drop_indices)

	# Drop malformed rows
	df = df.drop(index=drop_indices).reset_index(drop=True)

	total_after_drop = len(df)
	total_dropped = nan_count + malformed_count

	# Build lists
	entity1_list = []
	relation_list = []
	entity2_list = []
	simple_sentences_list = df['Simple_Sentence'].tolist()
	metadata_list = df['Metadata'].tolist()

	for triple in df['Triple']:
		parts = [p.strip().strip('"') for p in str(triple).split(',')]
		entity1_list.append(parts[0])
		relation_list.append(parts[1])
		entity2_list.append(parts[2])

	parsed_lists = {
		'Entity1': entity1_list,
		'Relation': relation_list,
		'Entity2': entity2_list,
		'Simple_Sentence': simple_sentences_list,
		'Metadata': metadata_list
	}

	return parsed_lists, total_before_drop, total_after_drop, nan_count, malformed_count

def main(disease_csv, triples_folder, start, limit=None):
	disease_names = read_disease_names(disease_csv, start, limit)
	stats_list = []
	parsed_data = {}

	logger = logging.getLogger("tagger")
	for disease in disease_names:
		triples_file = os.path.join(triples_folder, f"triples_{disease}.csv")
		parsed_lists, total_before, total_after, nan_count, malformed_count = process_triples_file(triples_file)

		if parsed_lists is None:
			parsed_lists = {
				'Entity1': [],
				'Relation': [],
				'Entity2': [],
				'Simple_Sentence': [],
				'Metadata': []
			}

		parsed_data[disease] = parsed_lists

		total_dropped = nan_count + malformed_count
		stats_list.append({
			'disease_name': disease,
			'total_before_drop': total_before,
			'total_after_drop': total_after,
			'total_dropped': total_dropped,
			'nan_count': nan_count,
			'malformed_count': malformed_count
		})

		logger.info(f"done for {disease}, dropped {total_dropped} triples ({nan_count} NaN + {malformed_count} malformed)")

	# Save statistics CSV
	stats_df = pd.DataFrame(stats_list)
	#stats_df.to_csv("triples_stats.csv", index=False)
	#print("Statistics saved to triples_stats.csv")

	return parsed_data, stats_df


def extract_response(input_string):
	pattern = r'### Response:\n(.*?)(?=\n###|$)'  # Updated pattern to ensure it captures the full response
	match = re.search(pattern, input_string, re.DOTALL)
	if match:
		response = match.group(1).strip()  # Extract and clean the response
		return response

def make_prompts(chunks):
	given_prompt = tagging_triples_prompt_with_relevant_sentences
	alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

		### Instruction:
		{}

		### Input:
		{}

		### Response:
		{}"""
	return [alpaca_prompt.format(given_prompt, chunk, "") for chunk in chunks]


def extract_tags(triple_str):
	triple_str = triple_str.strip()
	if triple_str.startswith("(") and triple_str.endswith(")"):
		triple_str = triple_str[1:-1].strip()
	
	parts = triple_str.split(",")
	if len(parts) != 3:
		return None, None 
	e1, _, e2 = [p.strip() for p in parts]
	tag1 = re.search(r'\((.*?)\)', e1).group(1) if re.search(r'\((.*?)\)', e1) else None
	tag2 = re.search(r'\((.*?)\)', e2).group(1) if re.search(r'\((.*?)\)', e2) else None

	return tag1, tag2

def tagger_for_triples(disease_csv, parsed_data, start, limit=None):
	"""
	For each disease, run tagging on triples (Entity1, Entity2, Sentence)
	using a vLLM backend, while preserving metadata and handling errors.
	"""
	disease_names = read_disease_names(disease_csv, start, limit)
	all_results = {}

	logger = logging.getLogger("tagger")
	for d_name in disease_names:
		responses_list = []
		if d_name not in parsed_data:
			logger.warning(f"Disease {d_name} not found in parsed data.")
			continue

		logger.info(f"Processing disease: {d_name}")

		# Extract the lists for this disease
		parsed_lists = parsed_data[d_name]
		entities1 = parsed_lists['Entity1']
		relations = parsed_lists['Relation']
		entities2 = parsed_lists['Entity2']
		sentences = parsed_lists['Simple_Sentence']
		metadatas = parsed_lists['Metadata']

		# Skip if there are no triples to process
		if len(entities1) == 0:
			logger.info(f"Skipping {d_name}: no triples to process.")
			continue

		disease_results = []

		chunks_this_disease = []
		for e1, rel, e2, sent in zip(entities1, relations, entities2, sentences):
			chunk_here = f"Entity1: {e1}, Relation: {rel}, Entity2: {e2}, Sentence: {sent}"
			chunks_this_disease.append(chunk_here)

		prompts = make_prompts(chunks_this_disease)

		batch_size = global_batch_size
		for i in range(0, len(prompts), batch_size):
			batch_inputs = prompts[i:i+batch_size]

			sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
			outputs = model.generate(batch_inputs, sampling_params)
			for output in outputs:
				full_text = output.prompt + output.outputs[0].text
				res = extract_response(full_text)
				responses_list.append(res)
		
		for i in range(len(responses_list)):
			tag1, tag2 = extract_tags(responses_list[i]) 
			result_entry = {
				'Entity1': entities1[i],
				'Relation': relations[i],
				'Entity2': entities2[i],
				'Simple_Sentence': sentences[i],
				'Metadata': metadatas[i],
				'Tagging_Response': responses_list[i],
				'Entity1_tag': tag1,
				'Entity2_tag': tag2,
				'Relation_tag': categorize_relation(relations[i])
			}
			# print(result_entry)
			disease_results.append(result_entry)

		all_results[d_name] = disease_results

		# Specify the columns you want in order
		columns = [
			"Entity1", "Entity1_tag",
			"Relation", "Relation_tag",
			"Entity2", "Entity2_tag",
			"Simple_Sentence", "Metadata"
		]

		# Convert the list of dicts to a DataFrame
		df = pd.DataFrame(disease_results)
		df = df[columns]

		# Save to CSV
		output_file = os.path.join("tagged_results_1", f"{d_name}_tagged_triples.csv")
		df.to_csv(output_file, index=False, encoding="utf-8")
		logger.info(f"Processed disease: {d_name}")

	return all_results


if __name__ == "__main__":
	# Step 1: Parse command-line arguments
	import argparse
	parser = argparse.ArgumentParser(description="Tagger for triples: tag and categorize biomedical triples")
	parser.add_argument(
		"-d", "--disease_csv",
		dest="disease_csv",
		default=None,
		help="Path to CSV file containing disease names (default from config or 5094_disease_folders_names.csv)"
	)
	parser.add_argument(
		"-t", "--triples_folder",
		dest="triples_folder",
		default=None,
		help="Folder containing triples CSVs (default from config or collected_results_1)"
	)
	parser.add_argument(
		"-b", "--batch_size",
		dest="batch_size",
		type=int,
		default=None,
		help="Batch size for LLM inference (default from config or 100)"
	)
	args = parser.parse_args()

	# Step 2: Load configuration from model_config.json
	model_config = load_model_config(os.path.join(os.getcwd(), "model_config.json"))

	# Step 3: Set file paths and batch size from args, config, or defaults
	disease_csv = args.disease_csv or model_config.get("disease_csv", "5094_disease_folders_names.csv")
	triples_folder = args.triples_folder or model_config.get("triples_folder", "collected_results_1")
	global_batch_size = args.batch_size or model_config.get("batch_size", 100)

	# Step 4: Initialize logger
	log_file = model_config.get("log_file", "tagger_log.log")
	init_logger(log_file, level=logging.INFO)

	# Step 5: Parse and clean triples for all diseases, collecting statistics
	parsed_data, stats_df = main(disease_csv, triples_folder, start=0, limit=None)

	# Step 6: Load the LLM model (Gemma or compatible) using config
	model = load_gemma_from_config(model_config)

	# Step 7: Load the sentence transformer and precompute category embeddings
	model_sentence_transformers, category_embeddings = load_model_sentence_transformer(model_config)

	# Step 8: Run the tagging pipeline for all diseases
	tagged_data = tagger_for_triples(disease_csv, parsed_data, start=0, limit=None)