attr-lgxa:
	python run_attribution_pipeline.py -a 1 -m 1 -d 0
	python run_attribution_pipeline.py -a 1 -m 1 -d 2
	python run_attribution_pipeline.py -a 1 -m 1 -d 4 -ent 1
	python run_attribution_pipeline.py -a 1 -m 1 -d 6
	python run_attribution_pipeline.py -a 1 -m 1 -d 6 -ent 1
	python run_attribution_pipeline.py -a 1 -m 2 -d 0
	python run_attribution_pipeline.py -a 1 -m 2 -d 2
	python run_attribution_pipeline.py -a 1 -m 2 -d 4 -ent 1
	python run_attribution_pipeline.py -a 1 -m 2 -d 6
	python run_attribution_pipeline.py -a 1 -m 2 -d 6 -ent 1
	python run_attribution_pipeline.py -a 1 -m 0 -d 0
	python run_attribution_pipeline.py -a 1 -m 0 -d 2
	python run_attribution_pipeline.py -a 1 -m 0 -d 4 -ent 1
	python run_attribution_pipeline.py -a 1 -m 0 -d 6
	python run_attribution_pipeline.py -a 1 -m 0 -d 6 -ent 1

attr-gradcam:
	python run_attribution_pipeline.py -a 3 -m 1 -d 0
	python run_attribution_pipeline.py -a 3 -m 1 -d 2
	python run_attribution_pipeline.py -a 3 -m 1 -d 4 -ent 1
	python run_attribution_pipeline.py -a 3 -m 1 -d 6
	python run_attribution_pipeline.py -a 3 -m 1 -d 6 -ent 1
	python run_attribution_pipeline.py -a 3 -m 2 -d 0
	python run_attribution_pipeline.py -a 3 -m 2 -d 2
	python run_attribution_pipeline.py -a 3 -m 2 -d 4 -ent 1
	python run_attribution_pipeline.py -a 3 -m 2 -d 6
	python run_attribution_pipeline.py -a 3 -m 2 -d 6 -ent 1
	python run_attribution_pipeline.py -a 3 -m 0 -d 0
	python run_attribution_pipeline.py -a 3 -m 0 -d 2
	python run_attribution_pipeline.py -a 3 -m 0 -d 4 -ent 1
	python run_attribution_pipeline.py -a 3 -m 0 -d 6
	python run_attribution_pipeline.py -a 3 -m 0 -d 6 -ent 1

attr-lig-rob:
	python run_attribution_pipeline.py -m 2 -d 0
	python run_attribution_pipeline.py -m 2 -d 2
	python run_attribution_pipeline.py -m 2 -d 4 -ent 1
	python run_attribution_pipeline.py -m 2 -d 6
	python run_attribution_pipeline.py -m 2 -d 6 -ent 1

attr-lig-ele:
	python run_attribution_pipeline.py -m 1 -d 0
	python run_attribution_pipeline.py -m 1 -d 2
	python run_attribution_pipeline.py -m 1 -d 4 -ent 1
	python run_attribution_pipeline.py -m 1 -d 6
	python run_attribution_pipeline.py -m 1 -d 6 -ent 1

attr-lgxa-ele:
	python run_attribution_pipeline.py -a 1 -m 1 -d 0
	python run_attribution_pipeline.py -a 1 -m 1 -d 2
	python run_attribution_pipeline.py -a 1 -m 1 -d 4 -ent 1
	python run_attribution_pipeline.py -a 1 -m 1 -d 6
	python run_attribution_pipeline.py -a 1 -m 1 -d 6 -ent 1

eval-lig-ele-inc:
	python run_pipeline.py -m 1 -d 0
	python run_pipeline.py -m 1 -d 2
	python run_pipeline.py -m 1 -d 4 -ent 1
	python run_pipeline.py -m 1 -d 6
	python run_pipeline.py -m 1 -d 6 -ent 1

eval-lig-ele-exc:
	python run_pipeline.py -m 1 -d 0 --exclude
	python run_pipeline.py -m 1 -d 2 --exclude
	python run_pipeline.py -m 1 -d 4 -ent 1 --exclude
	python run_pipeline.py -m 1 -d 6 --exclude
	python run_pipeline.py -m 1 -d 6 -ent 1 --exclude

eval-lgxa-ele:
	python run_pipeline.py -a 1 -m 1 -d 0 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 2 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 6 -ent 1 --exclude ${EXCLUDE}

eval-gradcam-ele:
	python run_pipeline.py -a 3 -m 1 -d 0 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 2 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 6 -ent 1 --exclude ${EXCLUDE}

eval-lig-rob:
	python run_pipeline.py -m 2 -d 0 --exclude ${EXCLUDE}
	python run_pipeline.py -m 2 -d 2 --exclude ${EXCLUDE}
	python run_pipeline.py -m 2 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -m 2 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -m 2 -d 6 -ent 1 --exclude ${EXCLUDE}

eval-lgxa-rob:
	python run_pipeline.py -a 1 -m 2 -d 0 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 2 -d 2 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 2 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 2 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 2 -d 6 -ent 1 --exclude ${EXCLUDE}

eval-gradcam-rob:
	python run_pipeline.py -a 3 -m 2 -d 0 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 2 -d 2 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 2 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 2 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 2 -d 6 -ent 1 --exclude ${EXCLUDE}

eval-ele-rest:
	python run_pipeline.py -a 0 -m 1 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 0 -m 1 -d 6 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 6 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 6 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -a 0 -m 1 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -a 1 -m 1 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -a 3 -m 1 -d 6 --exclude ${EXCLUDE}

eval-rob-rest:
	python run_pipeline.py -a 1 -m 2 -d 6 -ent 1
	python run_pipeline.py -a 1 -m 2 -d 6 -ent 1 --exclude true
	python run_pipeline.py -a 1 -m 2 -d 6 -ent 0
	python run_pipeline.py -a 1 -m 2 -d 6 -ent 0 --exclude true

qual-viz:
# 	python generate_qualitative_viz.py -m 2 -d 0 -doc 20235 -ref1 312 -ref2 241
# 	python generate_qualitative_viz.py -m 2 -d 0 -doc 12601 -ref1 28 -ref2 21
# 	python generate_qualitative_viz.py -m 1 -d 2 -doc 9869602 -ref1 226 -ref2 248
# 	python generate_qualitative_viz.py -m 1 -d 2 -doc 9856499 -ref1 83 -ref2 99
# 	python generate_qualitative_viz.py -m 2 -d 2 -doc 9450866 -ref1 34 -ref2 33
# 	python generate_qualitative_viz.py -m 2 -d 2 -doc 9689113 -ref1 191 -ref2 156
# 	python generate_qualitative_viz.py -m 1 -d 4 -ent 1 -doc Trastuzumab -ref1 55 -ref2 59
# 	python generate_qualitative_viz.py -m 1 -d 4 -ent 1 -doc 21868520 -ref1 67 -ref2 75
	python generate_qualitative_viz.py -m 2 -d 4 -ent 1 -doc Pseudoephedrine -ref1 47 -ref2 34
	python generate_qualitative_viz.py -m 2 -d 4 -ent 1 -doc Methotrimeprazine -ref1 189 -ref2 170
	python generate_qualitative_viz.py -m 1 -d 6 -ent 1 -doc 76 -ref1 142 -ref2 145
	python generate_qualitative_viz.py -m 1 -d 6 -ent 1 -doc 587 -ref1 59 -ref2 56
	python generate_qualitative_viz.py -m 2 -d 6 -ent 1 -doc 335 -ref1 38 -ref2 37
	python generate_qualitative_viz.py -m 2 -d 6 -ent 0 -doc 249 -ref1 112 -ref2 107

train-all: train-disease train-drug cleanup

train-disease:
	# bc5cdr
	python finetune_model.py -m 0 -d 0 -b 8 -e 8
	python finetune_model.py -m 1 -d 0 -b 8 -e 5 -l 0
	python finetune_model.py -m 2 -d 0 -b 8 -e 5 -l 0
	# euadr
	python finetune_model.py -m 0 -d 1 -b 16 -e 20
	python finetune_model.py -m 0 -d 1 -b 8 -e 10
	python finetune_model.py -m 1 -d 1 -b 16 -e 5 -l 0
	python finetune_model.py -m 2 -d 1 -b 16 -e 5 -l 0
	# NCBI
	python finetune_model.py -m 0 -d 2 -b 16 -e 20
	python finetune_model.py -m 1 -d 2 -b 16 -e 5 -l 0
	python finetune_model.py -m 2 -d 2 -b 16 -e 5 -l 0
	# SCAI
	python finetune_model.py -m 0 -d 3 -b 16 -e 20
	python finetune_model.py -m 0 -d 3 -b 8 -e 10
	python finetune_model.py -m 1 -d 3 -b 16 -e 5 -l 0
	python finetune_model.py -m 2 -d 3 -b 16 -e 5 -l 0

train-drug:
	# euadr
	python finetune_model.py -m 0 -d 1 -b 16 -e 20 -ent 1
	python finetune_model.py -m 0 -d 1 -b 8 -e 10 -ent 1
	python finetune_model.py -m 1 -d 1 -b 16 -e 5 -l 0 -ent 1
	python finetune_model.py -m 2 -d 1 -b 8 -e 10 -ent 1
	# MLEE
	python finetune_model.py -m 0 -d 5 -b 16 -e 20 -ent 1
	python finetune_model.py -m 0 -d 5 -b 8 -e 10 -ent 1
	python finetune_model.py -m 1 -d 5 -b 16 -e 5 -l 0 -ent 1
	python finetune_model.py -m 2 -d 5 -b 2 -e 10 -ent 1
	# DDI
	python finetune_model.py -m 0 -d 4 -b 16 -e 20 -ent 1
	python finetune_model.py -m 0 -d 4 -b 8 -e 10 -ent 1
	python finetune_model.py -m 1 -d 4 -b 16 -e 5 -l 0 -ent 1
	python finetune_model.py -m 2 -d 4 -b 2 -e 10 -ent 1

train-roberta-disease:
	python finetune_model.py -m 2 -d 0 -b 6 -e 10
	python finetune_model.py -m 2 -d 1 -b 4 -e 6
	python finetune_model.py -m 2 -d 1 -b 4 -e 10
	python finetune_model.py -m 2 -d 1 -b 6 -e 10
	python finetune_model.py -m 2 -d 2 -b 4 -e 6
	python finetune_model.py -m 2 -d 2 -b 4 -e 10
	python finetune_model.py -m 2 -d 2 -b 6 -e 10
	python finetune_model.py -m 2 -d 3 -b 4 -e 6
	python finetune_model.py -m 2 -d 3 -b 4 -e 10
	python finetune_model.py -m 2 -d 3 -b 6 -e 10

train-all-cadec: # single gpu
	python finetune_model.py -m 0 -d 6 -b 32 -e 20 -ent 1
	python finetune_model.py -m 0 -d 6 -b 16 -e 10 -ent 1
	python finetune_model.py -m 1 -d 6 -b 32 -e 5 -l 0 -ent 1
	python finetune_model.py -m 1 -d 6 -b 16 -e 5 -l 0 -ent 1
	python finetune_model.py -m 2 -d 6 -b 6 -e 10 -ent 1
	python finetune_model.py -m 2 -d 6 -b 8 -e 10 -ent 1
	python finetune_model.py -m 2 -d 6 -b 10 -e 10 -ent 1

train-all-cadec-alt-settings: # single gpu
	python finetune_model.py -m 0 -d 6 -b 32 -e 8 -ent 1
	python finetune_model.py -m 0 -d 6 -b 16 -e 8 -ent 1
	python finetune_model.py -m 1 -d 6 -b 24 -e 5 -l 0 -ent 1
	python finetune_model.py -m 2 -d 6 -b 6 -e 8 -ent 1
	python finetune_model.py -m 2 -d 6 -b 8 -e 10 -ent 1
	python finetune_model.py -m 2 -d 6 -b 4 -e 10 -ent 1

train-all-cadec-2gpu: # single gpu
	python finetune_model.py -m 0 -d 6 -b 16 -e 20 -ent 1
	python finetune_model.py -m 0 -d 6 -b 8 -e 10 -ent 1
	python finetune_model.py -m 1 -d 6 -b 16 -e 5 -l 0 -ent 1
	python finetune_model.py -m 1 -d 6 -b 8 -e 5 -l 0 -ent 1
	python finetune_model.py -m 2 -d 6 -b 3 -e 10 -ent 1
	python finetune_model.py -m 2 -d 6 -b 4 -e 10 -ent 1
	python finetune_model.py -m 2 -d 6 -b 5 -e 10 -ent 1

cleanup:
	cd trained_models
	find . -type d -name 'checkp*' -prune -exec rm -rf {} \;
	cd ..