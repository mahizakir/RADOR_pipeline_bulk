import pandas as pd
import torch
import re, csv, os, sys
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer, util
import warnings
from global_prompts import global_output_max_new_tokens, tagging_triples_prompt_with_relevant_sentences

# Function to print and append only selected messages
def log_to_file(*args, file_path="tagger_log.log"):
    with open(file_path, "a") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(*args)  # handle multiple arguments like print()
        sys.stdout = old_stdout


def load_gemma(max_model_len):
    # Suppress the warning related to slow image processor
    warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_length = global_output_max_new_tokens
    dtype = None
    load_in_4bit = True  
    use_gpu = True

    # Load model
    if use_gpu:
        model = LLM(
            model='unsloth/gemma-3-27b-it-unsloth-bnb-4bit',
            max_model_len=9000,
            quantization="bitsandbytes",
            gpu_memory_utilization=0.90,
            tensor_parallel_size=1,

        )

    return model


def load_model_sentence_transformers():
    model_sentence_transformers = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    base_categories_for_relations = {
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

    # Precompute embeddings for all examples
    category_embeddings = {cat: model_sentence_transformers.encode(examples) for cat, examples in base_categories_for_relations.items()}

    return model_sentence_transformers, category_embeddings


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

        print(f"done for {disease}, dropped {total_dropped} triples ({nan_count} NaN + {malformed_count} malformed)")

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

    for d_name in disease_names:
        responses_list = []
        if d_name not in parsed_data:
            print(f"Warning: Disease {d_name} not found in parsed data.")
            continue

        print(f"\nProcessing disease: {d_name}")

        # Extract the lists for this disease
        parsed_lists = parsed_data[d_name]
        entities1 = parsed_lists['Entity1']
        relations = parsed_lists['Relation']
        entities2 = parsed_lists['Entity2']
        sentences = parsed_lists['Simple_Sentence']
        metadatas = parsed_lists['Metadata']

        # Skip if there are no triples to process
        if len(entities1) == 0:
            print(f"Skipping {d_name}: no triples to process.")
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
        log_to_file("Processed disease: " + d_name)

    return all_results



if __name__ == "__main__":
    disease_csv = "5094_disease_folders_names.csv"
    triples_folder = "collected_results_1"

    parsed_data, stats_df = main(disease_csv, triples_folder, start=0, limit=None) 

    model = load_gemma(max_model_len=9000)
    model_sentence_transformers, category_embeddings = load_model_sentence_transformers()

    global_batch_size = 100
    tagged_data = tagger_for_triples(disease_csv, parsed_data, start=0, limit=None) 
