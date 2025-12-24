import os
import logging
import warnings
import torch
from vllm import LLM

def load_model_config(path):
	"""Load model configuration from a JSON file."""
	import json
	if not os.path.isfile(path):
		return {}
	try:
		with open(path, "r") as f:
			cfg = json.load(f)
		logging.getLogger(__name__).info("Loaded model config from %s", path)
		return cfg
	except Exception as e:
		logging.getLogger(__name__).warning("Could not load model config from %s: %s", path, e)
		return {}

def load_gemma_from_config(model_config):
	"""Load the Gemma LLM model using a config dict."""
	warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
	warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA device not available for Gemma model.")
	return LLM(
		model=model_config.get('llm_model_name', 'unsloth/gemma-3-27b-it-unsloth-bnb-4bit'),
		max_model_len=model_config.get('llm_max_model_len', 9000),
		quantization=model_config.get('quantization', 'bitsandbytes'),
		gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.90),
		tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
	)

def load_model_sentence_transformer(model_config):
	"""Load sentence transformer and precompute category embeddings."""
	model_name = model_config.get("sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2")
	model = SentenceTransformer(model_name)
	category_embeddings = {cat: model.encode(examples) for cat, examples in BASE_CATEGORIES_FOR_RELATIONS.items()}
	return model, category_embeddings