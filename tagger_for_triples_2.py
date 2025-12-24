import os
import re
import logging
import pandas as pd

from vllm import SamplingParams
from sentence_transformers import util
from global_prompts import global_output_max_new_tokens, tagging_triples_prompt_with_relevant_sentences
from utils.model_utils import load_model_config, load_gemma_from_config, load_sentence_transformer


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

def log_to_file(logger, *args):
	"""Log messages using the provided logger."""
	logger.info(" ".join(str(a) for a in args))

BASE_CATEGORIES_FOR_RELATIONS = {
	"Classification/Type": [
		"IS", "ARE", "MATCHED TO", "IDENTIFIED AS", "CLASSIFIED AS", "CATEGORIZED AS", 
		"REPRESENTS", "DESCRIBED AS", "TYPIFIED BY", "DEFINED AS"
	],
	"Containment/Component": [
		"HAS", "HAVE", "CONTAINS", "INCLUDE", "PART OF", "COMPOSED OF", "ENCLOSED IN", 
		"HOUSED IN", "CONSTITUTES", "IS A COMPONENT OF", "IS PART OF", "BELONGS TO"
	],
	"Location/Presence": [
		"PRESENT IN", "LOCATED IN", "FOUND IN", "RESIDES IN", "EXISTS IN", 
		"OCCURS IN", "SITUATED IN", "DETECTED IN", "APPEARS IN", "EXHIBITS"
	],
	"Cause/Effect": [
		"CAUSES", "RESULTS IN", "LEADS TO", "INDUCES", "CONTRIBUTES TO", 
		"PROMOTES", "TRIGGERS", "PROVOKES", "FACILITATES", "ENCOURAGES"
	],
	"Function/Usage": [
		"USED IN", "APPLIED TO", "PERFORMED IN", "EMPLOYED FOR", "UTILIZED IN", 
		"DESIGNED FOR", "FUNCTIONS AS", "SERVES TO", "ACTS AS", "PURPOSED FOR"
	],
	"Measurement/Expression": [
		"MEASURED IN", "SHOWS", "HAS", "DEMONSTRATES", "EXHIBITS", 
		"DISPLAYS", "INDICATES", "QUANTIFIED BY", "REPRESENTS", "SIGNIFIES"
	],
	"Association/Link": [
		"ASSOCIATED WITH", "LINK", "RELATED TO", "CORRELATED WITH", "CONNECTED TO", 
		"INTERACTS WITH", "ALIGNED WITH", "BONDS WITH", "TIES TO", "INFLUENCES"
	],
	"Process/Progression": [
		"OCCURS IN", "PROGRESSES TO", "FORMS", "DEVELOPS INTO", "TRANSITIONS TO", 
		"EVOLVES TO", "MATURES INTO", "ADVANCES TO", "CHANGES INTO", "MOVES TO"
	],
	"Documentation/Evidence": [
		"DOCUMENTED IN", "ANALYZED IN", "REPORTED FOR", "OBSERVED IN", "RECORDED IN", 
		"VALIDATED IN", "CONFIRMED IN", "ESTABLISHED IN", "NOTED IN", "REGISTERED IN"
	],
	"Action/Delivery": [
		"DELIVERS", "TARGETS", "TRANSPORTS TO", "CARRIES TO", "DISTRIBUTES TO", 
		"DISPATCHES TO", "CONVEYS", "SUPPLIES", "PROVIDES TO", "ENFOLDS"
	],
	"Similarity/Comparison": [
		"SIMILAR TO", "COMPARED TO", "MATCHES", "ANALOGOUS TO", "EQUIVALENT TO", 
		"RESEMBLES", "ALIGNED WITH", "COMPARABLE TO", "LIKE", "PARALLELS"
	],
	"Activation/Inhibition": [
		"ACTIVATED IN", "INHIBITED BY", "STIMULATED BY", "INITIATED BY", "INFLUENCED BY", 
		"ENHANCED BY", "SUPPRESSED BY", "MODULATED BY", "TRIGGERED BY", "REGULATED BY"
	],
	"Regulation/Control": [
		"REGULATED BY", "CONTROLS", "SUPPRESSED BY", "MODULATED BY", "GOVERNED BY", 
		"MAINTAINED BY", "ADJUSTED BY", "DIRECTED BY", "OVERSEEN BY", "ADMINISTERED BY"
	],
	"Interaction/Engagement": [
		"INTERACTS WITH", "ENGAGES WITH", "COOPERATES WITH", "COLLABORATES WITH", 
		"PARTICIPATES IN", "CONTRIBUTES TO", "ASSISTS WITH", "SUPPORTS", "REINFORCES", "CONNECTS TO"
	],
	"Development/Change": [
		"EVOLVES", "CHANGES", "TRANSFORMS", "ADAPTS", "MODIFIES", 
		"MATURES", "GROWS", "PROGRESSES", "DEVELOPS", "REFINES"
	],
	"Condition/State": [
		"IS PRESENT", "IS ABSENT", "EXISTS AS", "OCCURS AS", "MANIFESTS AS", 
		"APPEARS AS", "IS DETERMINED BY", "DEPENDS ON", "AFFECTS", "IMPACTS"
	],
	"Therapeutic/Treatment": [
		"TREATED BY", "MANAGED BY", "ADDRESSING", "HELPED BY", "CURING", 
		"FACILITATES TREATMENT", "ENHANCES HEALING", "ASSISTS IN RECOVERY", 
		"RESTORES", "SUPPORTS RECOVERY"
	],
	"Mechanism/Pathway": [
		"INVOLVED IN", "ENGAGED IN", "PARTICIPATES IN", "PLAYS A ROLE IN", 
		"CONTRIBUTES TO", "FACILITATES", "ENABLES", "AFFECTS", "MODIFIES", "INTERFERES WITH"
	]
}


def categorize_relation(rel, model_sentence_transformers, category_embeddings):
	"""Categorize a relation string using sentence transformer embeddings."""
	rel_embed = model_sentence_transformers.encode(rel)
	best_cat, best_score = None, float("-inf")
	for cat, cat_embeds in category_embeddings.items():
		score = util.cos_sim(rel_embed, cat_embeds).max().item()
		if score > best_score:
			best_cat, best_score = cat, score
	return best_cat


###############################################################################
# STEP 7: READ DISEASE NAMES FROM CSV
###############################################################################
def read_disease_names(disease_csv, start=0, limit=None):
	"""Read disease names from a CSV file."""
	df = pd.read_csv(disease_csv, header=None)
	names = df[0].tolist()[start:]
	if limit:
		names = names[:limit]
	return names


###############################################################################
# STEP 8: READ AND CLEAN TRIPLES FILE
###############################################################################
def process_triples_file(triples_file):
	"""Read and clean a triples CSV, parse Entity1, Relation, Entity2."""
	if not os.path.exists(triples_file):
		return None, 0, 0, 0, 0
	df = pd.read_csv(triples_file)
	total_before_drop = len(df)
	nan_count = df['Triple'].isna().sum()
	df = df.dropna(subset=['Triple']).reset_index(drop=True)
	drop_indices = [idx for idx, triple in enumerate(df['Triple']) if len([p.strip().strip('"') for p in str(triple).split(',')]) != 3]
	malformed_count = len(drop_indices)
	df = df.drop(index=drop_indices).reset_index(drop=True)
	total_after_drop = len(df)
	entity1_list, relation_list, entity2_list = [], [], []
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

###############################################################################
# STEP 9: MAIN TRIPLES FILE PROCESSING FOR ALL DISEASES
###############################################################################
def main(disease_csv, triples_folder, start=0, limit=None, logger=None):
	"""Main function to process all triples files for diseases."""
	disease_names = read_disease_names(disease_csv, start, limit)
	stats_list = []
	parsed_data = {}
	for disease in disease_names:
		triples_file = os.path.join(triples_folder, f"triples_{disease}.csv")
		parsed_lists, total_before, total_after, nan_count, malformed_count = process_triples_file(triples_file)
		if parsed_lists is None:
			parsed_lists = {
				'Entity1': [], 'Relation': [], 'Entity2': [], 'Simple_Sentence': [], 'Metadata': []
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
		if logger:
			logger.info(f"done for {disease}, dropped {total_dropped} triples ({nan_count} NaN + {malformed_count} malformed)")
		else:
			print(f"done for {disease}, dropped {total_dropped} triples ({nan_count} NaN + {malformed_count} malformed)")
	stats_df = pd.DataFrame(stats_list)
	return parsed_data, stats_df


###############################################################################
# STEP 10: EXTRACT RESPONSE FROM LLM OUTPUT
###############################################################################
def extract_response(input_string):
	"""Extract the response from an Alpaca-formatted string."""
	pattern = r'### Response:\n(.*?)(?=\n###|$)'
	match = re.search(pattern, input_string, re.DOTALL)
	if match:
		return match.group(1).strip()

###############################################################################
# STEP 11: MAKE PROMPTS FOR LLM
###############################################################################
def make_prompts(chunks):
	"""Create Alpaca-formatted prompts for each chunk."""
	given_prompt = tagging_triples_prompt_with_relevant_sentences
	alpaca_prompt = (
		"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
		"### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
	)
	return [alpaca_prompt.format(given_prompt, chunk, "") for chunk in chunks]


###############################################################################
# STEP 12: EXTRACT TAGS FROM TRIPLE STRING
###############################################################################
def extract_tags(triple_str):
	"""Extract tags from a triple string."""
	triple_str = triple_str.strip()
	if triple_str.startswith("(") and triple_str.endswith(")"):
		triple_str = triple_str[1:-1].strip()
	parts = triple_str.split(",")
	if len(parts) != 3:
		return None, None
	e1, _, e2 = [p.strip() for p in parts]
	tag1 = re.search(r'\((.*?)\)', e1)
	tag2 = re.search(r'\((.*?)\)', e2)
	return (tag1.group(1) if tag1 else None, tag2.group(1) if tag2 else None)


###############################################################################
# STEP 13: TAGGER MAIN FUNCTION - TAG AND CATEGORIZE TRIPLES
###############################################################################
def tagger_for_triples(disease_csv, parsed_data, start=0, limit=None, model=None, model_sentence_transformers=None, category_embeddings=None, batch_size=100, output_dir="tagged_results_1", logger=None):
	"""Run tagging on triples for each disease using vLLM and sentence transformer."""
	disease_names = read_disease_names(disease_csv, start, limit)
	all_results = {}
	os.makedirs(output_dir, exist_ok=True)
	for d_name in disease_names:
		if d_name not in parsed_data:
			if logger:
				logger.warning(f"Disease {d_name} not found in parsed data.")
			else:
				print(f"Warning: Disease {d_name} not found in parsed data.")
			continue
		if logger:
			logger.info(f"Processing disease: {d_name}")
		else:
			print(f"\nProcessing disease: {d_name}")
		parsed_lists = parsed_data[d_name]
		entities1 = parsed_lists['Entity1']
		relations = parsed_lists['Relation']
		entities2 = parsed_lists['Entity2']
		sentences = parsed_lists['Simple_Sentence']
		metadatas = parsed_lists['Metadata']
		if not entities1:
			if logger:
				logger.info(f"Skipping {d_name}: no triples to process.")
			else:
				print(f"Skipping {d_name}: no triples to process.")
			continue
		chunks_this_disease = [f"Entity1: {e1}, Relation: {rel}, Entity2: {e2}, Sentence: {sent}" for e1, rel, e2, sent in zip(entities1, relations, entities2, sentences)]
		prompts = make_prompts(chunks_this_disease)
		responses_list = []
		for i in range(0, len(prompts), batch_size):
			batch_inputs = prompts[i:i+batch_size]
			sampling_params = SamplingParams(max_tokens=global_output_max_new_tokens)
			outputs = model.generate(batch_inputs, sampling_params)
			for output in outputs:
				full_text = output.prompt + output.outputs[0].text
				res = extract_response(full_text)
				responses_list.append(res)
		disease_results = []
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
				'Relation_tag': categorize_relation(relations[i], model_sentence_transformers, category_embeddings)
			}
			disease_results.append(result_entry)
		all_results[d_name] = disease_results
		columns = ["Entity1", "Entity1_tag", "Relation", "Relation_tag", "Entity2", "Entity2_tag", "Simple_Sentence", "Metadata"]
		df = pd.DataFrame(disease_results)[columns]
		output_file = os.path.join(output_dir, f"{d_name}_tagged_triples.csv")
		df.to_csv(output_file, index=False, encoding="utf-8")
		if logger:
			log_to_file(logger, "Processed disease: " + d_name)
		else:
			print("Processed disease: " + d_name)
	return all_results

###############################################################################
# STEP 14: ENTRYPOINT - ARGUMENT PARSING, CONFIG LOADING, PIPELINE EXECUTION
###############################################################################
if __name__ == "__main__":
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
	parser.add_argument(
		"-o", "--output_dir",
		dest="output_dir",
		default=None,
		help="Directory to save tagged results (default from config or tagged_results_1)"
	)
	parser.add_argument(
		"-l", "--log_file",
		dest="log_file",
		default=None,
		help="Log file path (default from config or tagger_log.log)"
	)
	args = parser.parse_args()

	# Load config
	model_config_path = os.path.join(os.getcwd(), "model_config.json")
	model_config = load_model_config(model_config_path)

	# Use args if provided, else fallback to config, else fallback to hardcoded default
	disease_csv = args.disease_csv or model_config.get("disease_csv", "5094_disease_folders_names.csv")
	triples_folder = args.triples_folder or model_config.get("triples_folder", "collected_results_1")
	batch_size = args.batch_size or model_config.get("batch_size", 100)
	output_dir = args.output_dir or model_config.get("output_dir", "tagged_results_1")
	log_file = args.log_file or model_config.get("log_file", "tagger_log.log")

	logger = init_logger(log_file, level=logging.INFO)

	parsed_data, stats_df = main(disease_csv, triples_folder, start=0, limit=None, logger=logger)
	model = load_gemma_from_config(model_config)
	model_sentence_transformers, category_embeddings = load_sentence_transformer()
	tagged_data = tagger_for_triples(
		disease_csv,
		parsed_data,
		start=0,
		limit=None,
		model=model,
		model_sentence_transformers=model_sentence_transformers,
		category_embeddings=category_embeddings,
		batch_size=batch_size,
		output_dir=output_dir,
		logger=logger
	)
