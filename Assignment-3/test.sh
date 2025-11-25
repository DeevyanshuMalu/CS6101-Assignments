unset HF_HUB_CACHE

curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
python test.py --checkpoint-to-load 2
python test.py --quantize-bits 8 --checkpoint-to-load 2
# curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
python test.py --unfreeze-bert --checkpoint-to-load 2
python test.py --unfreeze-bert --quantize-bits 8 --checkpoint-to-load 2
# curl --location-trusted -u 22b1256:459c74cf8ab1f998b33f94dcd0deada3 "https://internet-sso.iitb.ac.in/login.php"
python test.py --unfreeze-bert --doc-expansion --checkpoint-to-load 2
python test.py --unfreeze-bert --doc-expansion --quantize-bits 8 --checkpoint-to-load 2