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
	python run_pipeline.py -m 1 -d 0
	python run_pipeline.py -m 1 -d 2
	python run_pipeline.py -m 1 -d 4 -ent 1
	python run_pipeline.py -m 1 -d 6
	python run_pipeline.py -m 1 -d 6 -ent 1

eval-lig-rob:
	python run_pipeline.py -m 2 -d 0
	python run_pipeline.py -m 2 -d 2
	python run_pipeline.py -m 2 -d 4 -ent 1
	python run_pipeline.py -m 2 -d 6
	python run_pipeline.py -m 2 -d 6 -ent 1

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