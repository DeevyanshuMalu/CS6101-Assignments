unset HF_HUB_CACHE

# python train.py
python impact_scores.py --checkpoint-to-load 2
python impact_scores.py --quantize-bits 8 --checkpoint-to-load 2
# python train.py --unfreeze-bert
python impact_scores.py --unfreeze-bert --checkpoint-to-load 2
python impact_scores.py --unfreeze-bert --quantize-bits 8 --checkpoint-to-load 2
# python train.py --unfreeze-bert --doc-expansion
python impact_scores.py --unfreeze-bert --doc-expansion --checkpoint-to-load 2
python impact_scores.py --unfreeze-bert --doc-expansion --quantize-bits 8 --checkpoint-to-load 2
