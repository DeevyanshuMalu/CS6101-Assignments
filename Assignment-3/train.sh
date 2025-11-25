unset HF_HUB_CACHE

curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
# python train.py
python impact_scores.py --checkpoint-to-load 2
python impact_scores.py --quantize-bits 8 --checkpoint-to-load 2
curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
# python train.py --unfreeze-bert
python impact_scores.py --unfreeze-bert --checkpoint-to-load 2
python impact_scores.py --unfreeze-bert --quantize-bits 8 --checkpoint-to-load 2
curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
# python train.py --unfreeze-bert --doc-expansion
python impact_scores.py --unfreeze-bert --doc-expansion --checkpoint-to-load 2
python impact_scores.py --unfreeze-bert --doc-expansion --quantize-bits 8 --checkpoint-to-load 2