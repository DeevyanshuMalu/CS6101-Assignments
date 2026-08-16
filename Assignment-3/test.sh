unset HF_HUB_CACHE

python test.py --checkpoint-to-load 2
python test.py --quantize-bits 8 --checkpoint-to-load 2
python test.py --unfreeze-bert --checkpoint-to-load 2
python test.py --unfreeze-bert --quantize-bits 8 --checkpoint-to-load 2
python test.py --unfreeze-bert --doc-expansion --checkpoint-to-load 2
python test.py --unfreeze-bert --doc-expansion --quantize-bits 8 --checkpoint-to-load 2
