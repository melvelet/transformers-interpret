attr-pipe-all: attr-pipe-lig

attr-pipe-lig: attr-pipe-lig-disease attr-pipe-lig-drug

attr-pipe-lig-disease: attr-pipe-lig-disease-bc5cdr attr-pipe-lig-disease-ncbi attr-pipe-lig-disease-euadr

attr-pipe-lig-drug: attr-pipe-lig-drug-euadr attr-pipe-lig-drug-ddi

attr-pipe-lig-disease-bc5cdr:
	python run_attribution_pipeline.py -m 0 -d 0
	python run_attribution_pipeline.py -m 1 -d 0

attr-pipe-lig-disease-ncbi:
	python run_attribution_pipeline.py -m 0 -d 2
	python run_attribution_pipeline.py -m 1 -d 2

attr-pipe-lig-disease-euadr:
	python run_attribution_pipeline.py -m 0 -d 1
	python run_attribution_pipeline.py -m 1 -d 1

attr-pipe-lig-drug-euadr:
	python run_attribution_pipeline.py -m 0 -d 1 -ent 1
	python run_attribution_pipeline.py -m 1 -d 1 -ent 1

attr-pipe-lig-drug-ddi:
	python run_attribution_pipeline.py -m 0 -d 4 -ent 1
	python run_attribution_pipeline.py -m 1 -d 4 -ent 1

pipe-all: pipe-lig

pipe-lig: pipe-lig-disease pipe-lig-drug

pipe-lig-disease: pipe-lig-disease-bc5cdr pipe-lig-disease-ncbi pipe-lig-disease-euadr

pipe-lig-drug: pipe-lig-drug-euadr pipe-lig-drug-ddi

pipe-lig-disease-bc5cdr:
	python run_pipeline.py -m 0 -d 0
	python run_pipeline.py -m 1 -d 0

pipe-lig-disease-ncbi:
	python run_pipeline.py -m 0 -d 2
	python run_pipeline.py -m 1 -d 2

pipe-lig-disease-euadr:
	python run_pipeline.py -m 0 -d 1
	python run_pipeline.py -m 1 -d 1

pipe-lig-drug-euadr:
	python run_pipeline.py -m 0 -d 1 -ent 1
	python run_pipeline.py -m 1 -d 1 -ent 1

pipe-lig-drug-ddi:
	python run_pipeline.py -m 0 -d 4 -ent 1
	python run_pipeline.py -m 1 -d 4 -ent 1

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
	python finetune_model.py -m 2 -d 1 -b 2 -e 10 -ent 1
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

cleanup:
	cd trained_models
	find . -type d -name 'checkp*' -prune -exec rm -rf {} \;
	cd ..