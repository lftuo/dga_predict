#!/usr/bin/env bash
python tld_appender.py
python feat_n_gram_rank_extractor.py
echo "have you downloaded https://github.com/rrenaud/Gibberish-Detector ? put gib_detect_train.py and gib_model.pki here"
python feat_extractor.py
python feat_normalizer.py
python feat_vectorizer.py
python predict.py
