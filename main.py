# import
# from .dataset import list_subjects, load_raw, segment_raw
# from .analyses import decode, time_gen
# from .utils

import mne
import mne_bids
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import pandas as pd
from decode_gen_target import PATHS
from decode_gen_target import _decod_one_subject, correlate, _get_epochs
#from scipy.ndimage.interpolation import shift


### constants ###
name_data = "bids_anonym"
# name_data = "le_petit_prince"

nb_min_ses = 0
nb_max_ses = 2  # 2

nb_min_task = 0
nb_max_task = 4  # 4
###

def list_subjects():
    subjects = pd.read_csv(PATHS.bids / "participants0.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    return subjects

def setup_model(objective="regression"):
    if objective == "regression":
        alphas = np.logspace(-4, 4, 10)
        model = RidgeCV(alphas)
        scoring = correlate
    else:
        alphas = np.logspace(-4, 4, 10)
        model = RidgeClassifierCV(alphas)
        scoring = roc_auc_score
    return model, scoring


def decode(epochs, target):  # To reproduce figure from Gwillimas and al, Nature 2022.
    # To see if a model trained on a time step can predict above chance the future phonemes
    # from their respective neural activity. /!\ This informs us on the stability of neural representations, if they
    # are similarly decadable, but not on if some mental representations at time t already contains informations
    # on predictions of future phonemes/words (for this we would need to train the decoder accordingly,
    # on train set at time t and test set at time t+x)
    X = epochs.get_data()
    Y = epochs.metadata[target].values
    n_trials, n_channels, n_times = X.shape
    n_splits = 5
    cv = GroupKFold(n_splits)
    groups = epochs.metadata.sentence_id
    scores = []
    model, scoring = setup_model()
    for train, test in cv.split(X, groups=groups):
        for t in range(n_times):
            model.fit(X[train, :, t], Y[train])
            for t2 in range(n_times):
                Y_pred = model.predict(X[test, :, t2])
                scores_ = []
                for shift in range(-4, 5):
                    Y_ = Y.shift(shift)
                    score = scoring(Y_[test], Y_pred)
                    scores_.append(score)
                scores.append(scores_)
    return scores


# report = mne.Report()
report_TG = mne.Report()
#subjects = list_subjects()
subjects = ["02"]
ph_info = pd.read_csv(
    "phoneme_info.csv"
)  # phonation: "v", "uv", what do these mean ? (voiced ? as in ~ vowel)
targets = ["vowels"] #"phonemes", "words"
reg_classif = "regression"
model0, score_fct = setup_model(objective=reg_classif)

#for subject in subjects:
    #print("subject " + subject)

group_of_subjects = [["01"]] #[["01"], ["02"], ["04"], ["01", "02", "04"],  ["05"]]
for subjects in group_of_subjects:
    subject_str = ""
    for subject in subjects:
        subject_str = subject_str+ subject + "_"
    print(subject_str)
    # subject_str = subjects[0] #"01-04"
    epochs = _get_epochs(subjects, nb_min_task, nb_max_task, nb_min_ses, nb_max_ses)
    for target in targets:
        out = _decod_one_subject(
            report_TG, subject_str, target, epochs, nb_min_task, nb_max_task, nb_min_ses, nb_max_ses,
            model0=model0, score_fct=score_fct, bool_several_shadow = True
        )
        if out is None:
            continue
        report_TG.add_figure(out, subject_str, tags=str(target) + "-" + str(subject_str))
        # report.save("decoding.html", open_browser=False, overwrite=True)
        report_TG.save("decoding_TG_rainbow_s"+ subject_str +"wo_baseline__"+reg_classif+".html", open_browser=False, overwrite=True)
    print("done for subject " + subject_str)
