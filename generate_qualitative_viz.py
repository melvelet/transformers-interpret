import torch
from traitlets.config.loader import ArgumentParser

from sample_qualitative_viz import QualitativeVisualizer

dataset_names = [
    'bc5cdr',
    'euadr',
    'ncbi_disease',
    'scai_disease',
    'ddi_corpus',
    'mlee',
    'cadec',
]

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model_no", type=int)
parser.add_argument("-d", "--dataset", dest="dataset_no", type=int)
# parser.add_argument("-a", "--attribution-type", dest="attribution_type_no", type=int, default=0)
# parser.add_argument("-max", "--max-documents", dest="max_documents", type=int, default=0)
# parser.add_argument("-s", "--start-document", dest="start_document", type=int, default=0)
parser.add_argument("-k", "--k-value", dest="k_value", type=int, default=5)
parser.add_argument("-ent", "--entity", dest="entity", type=int, default=0)
parser.add_argument("-doc", "--doc-id", dest="doc_id", type=str, default='0')
parser.add_argument("-ref1", "--mod1-ref-token-idx", dest="mod1_ref_token_idx", type=int, default=0)
parser.add_argument("-ref2", "--mod2-ref-token-idx", dest="mod2_ref_token_idx", type=int, default=0)
parser.add_argument("-eval", "--eval", dest="eval", type=str, default='')
# parser.add_argument("-e", "--exclude", dest="exclude_reference_token", type=bool, default=False)
args = parser.parse_args()

huggingface_models = [1, 2] if args.model_no == 1 else [2, 1]
doc_id = args.doc_id
mod1_ref_token_idx = args.mod1_ref_token_idx
mod2_ref_token_idx = args.mod2_ref_token_idx
k_value = args.k_value
# eval = args.eval

viz = QualitativeVisualizer()
viz.load_dataset(dataset=args.dataset_no)
viz.load_tokenizers(models=huggingface_models)
viz.load_other_pipeline(base_path='./trained_models/')
viz.load_entities(base_path='./results/scores/')
viz.prepare(doc_id=doc_id, ref1_token_idx=mod1_ref_token_idx, ref2_token_idx=mod2_ref_token_idx)
with torch.no_grad():
    viz.ensure_attr_scores_in_other_model(mod2_ref_token_idx, k_value)
for collapse_threshold in [0.05, 0.02, 0.1]:
    latex_tables = viz.print_table(k_value=k_value, collapse_threshold=collapse_threshold)
    with open(f"viz/example_{'bioelectra' if args.model_no == 1 else 'roberta'}_{dataset_names[args.dataset_no]}_{doc_id}_{mod1_ref_token_idx}_{mod2_ref_token_idx}_{k_value}.txt", 'w+') as f:
        f.write(latex_tables)
