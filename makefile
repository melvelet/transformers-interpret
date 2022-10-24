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

eval-lig-ele:
	python run_pipeline.py -m 1 -d 0 --exclude ${EXCLUDE}
	python run_pipeline.py -m 1 -d 2 --exclude ${EXCLUDE}
	python run_pipeline.py -m 1 -d 4 -ent 1 --exclude ${EXCLUDE}
	python run_pipeline.py -m 1 -d 6 --exclude ${EXCLUDE}
	python run_pipeline.py -m 1 -d 6 -ent 1 --exclude ${EXCLUDE}

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
# 	python generate_qualitative_viz.py -m 2 -d 0 -doc 9591 -ref1 4 -ref2 4
# 	python generate_qualitative_viz.py -m 2 -d 0 -doc 17643 -ref1 20 -ref2 20
	python generate_qualitative_viz.py -m 1 -d 2 -doc 9724771 -ref1 314 -ref2 395
	python generate_qualitative_viz.py -m 1 -d 2 -doc 9950360 -ref1 55 -ref2 79
	python generate_qualitative_viz.py -m 2 -d 2 -doc 9580132 -ref1 68 -ref2 54
	python generate_qualitative_viz.py -m 2 -d 2 -doc 9856499 -ref1 1 -ref2 21
# 	python generate_qualitative_viz.py -m 1 -d 4 -ent 1 -doc Rituximab -ref1 16 -ref2 18
# 	python generate_qualitative_viz.py -m 1 -d 4 -ent 1 -doc Succimer -ref1 65 -ref2 56
# 	python generate_qualitative_viz.py -m 2 -d 4 -ent 1 -doc Medroxyprogesterone -ref1 124 -ref2 106
# 	python generate_qualitative_viz.py -m 2 -d 4 -ent 1 -doc Steptokinase -ref1 93 -ref2 73
# 	python generate_qualitative_viz.py -m 1 -d 6 -ent 1 -doc 76 -ref1 142 -ref2 145
# 	python generate_qualitative_viz.py -m 1 -d 6 -ent 1 -doc 75 -ref1 61 -ref2 96
# 	python generate_qualitative_viz.py -m 2 -d 6 -ent 1 -doc 358 -ref1 355 -ref2 337
# 	python generate_qualitative_viz.py -m 1 -d 2 -doc 9848786 -ref1 36 -ref2 41
# 	python generate_qualitative_viz.py -m 1 -d 2 -doc 9448273 -ref1 59 -ref2 67
# 	python generate_qualitative_viz.py -m 2 -d 2 -doc 941901 -ref1 16 -ref2 13
# 	python generate_qualitative_viz.py -m 2 -d 2 -doc 9590284 -ref1 5 -ref2 5

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