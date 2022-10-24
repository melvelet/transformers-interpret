from sample_qualitative_viz import QualitativeVisualizer
viz = QualitativeVisualizer()
viz.load_dataset(dataset=0)
huggingface_models = [1, 2]
# huggingface_models = [2, 1]
viz.load_tokenizers(huggingface_models)
viz.load_entities(base_path='./results/scores/', attributions=[0], entity_type=0)
ref1_token_idx = None
doc_id = None
allow_zero = True
eval_ = 'FP'
mod1_ref_token_idx, ref_token, doc_id1, doc_id2 = viz.pick_entities(eval_=eval_, n_value=1, doc_id=doc_id, ref1_token_idx=ref1_token_idx, allow_zero=allow_zero)

mod1_ref_token_idx, ref_token, doc_id1, doc_id2

mod2_ref_token_idx = mod1_ref_token_idx
viz.find_in_other_model(ref_token, reference_token_idx=mod2_ref_token_idx)