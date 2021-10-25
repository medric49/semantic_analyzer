import fasttext

import utils

utils.transform_tsv_to_fasttext_format('data/cola_public/raw/in_domain_train.tsv', 'train.txt')

utils.transform_tsv_to_fasttext_format('data/cola_public/raw/in_domain_dev.tsv', 'eval.txt')
