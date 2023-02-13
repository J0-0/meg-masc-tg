# import
# from .dataset import list_subjects, load_raw, segment_raw
# from .analyses import decode, time_gen
# from .utils

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import pandas as pd
from decode_gen_target import PATHS
from decode_gen_target import _decod_one_subject

def correlate(X, Y):
    if X.ndim == 1 :
        X = X[:, None]
    if Y.ndim == 1 :
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    SX2 = (X ** 2).sum(0) ** 0.5
    SY2 = (Y ** 2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    if (SX2 * SY2).any() != 0 :
        result = SXY / (SX2 * SY2)
    else :
        result = 0
    return result

def setup_model(objective='regression'):
    if objective == 'regression':
        alphas = np.logspace(-4, 4, 10)
        model = RidgeCV(alphas)
        scoring = correlate
    else:
        alphas = np.logspace(-4, 4, 10)
        model = RidgeClassifierCV(alphas)
        scoring = roc_auc_score
    return model, scoring

def list_subjects():
    subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x : x.split("-")[1]).values
    return subjects

def decode(epochs, target):
    X = epochs.get_data()
    Y = epochs.metadata[target].values
    n_trials, n_channels, n_times = X.shape
    n_splits = 5
    model, scoring = setup_model()
    cv = GroupKFold(n_splits)
    groups = epochs.metadata.sentence_id
    scores = []
    for train, test in cv.split(X, groups=groups):
        for t in range(n_times):
            model.fit(X[train, :, t], Y[train])
            for t2 in range(n_times) :
                Y_pred = model.predict(X[test, :, t2], Y[test])
                scores_ = []
                for shift in range(-4, 5):
                    Y_ = Y.shift(shift)
                    score = scoring(Y_[test], Y_pred)
                    scores_.append(score)
                scores.append(scores_)
    return scores

subjects = list_subjects()
ph_info = pd.read_csv("phoneme_info.csv")  # phonation: "v", "uv", what do these mean ? (voiced ? as in ~ vowel)
targets = ["vowels", "phonemes", "words"]
for target in targets:
    for subject in subjects:
        _decod_one_subject(subject, target)

