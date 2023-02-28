import mne
import mne_bids
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from tqdm import trange
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
import time
from sklearn.metrics import roc_auc_score
from scipy.ndimage.interpolation import shift


matplotlib.use("Agg")
mne.set_log_level(False)

### constants ###
names = ['v_v_lo_b_u_c', 'v_v_lo_f_u_f', 'v_v_m_b_r_f', 'v_v_m_f_u_f', 'v_o_l_f_u_n', 'uv_f_c_f_u_n', 'v_o_c_f_u_n',
          'v_f_d_f_u_n', 'v_a_v_b_u_f', 'v_v_h_f_u_f', 'uv_f_d_f_u_n', 'v_o_v_b_u_n', 'uv_f_g_b_u_n', 'v_f_c_f_u_n',
          'uv_o_v_b_u_n', 'v_a_c_f_u_n', 'v_n_l_f_u_n', 'v_n_c_f_u_n', 'v_n_v_b_u_n', 'v_v_h_b_r_f', 'uv_o_l_f_u_n',
          'v_a_v_b_u_n', 'uv_o_c_f_u_n', 'v_a_l_f_r_n'] # all categories of phonemes (24)

name_data = "bids_anonym"
further_before, further_after = -2, 4
n_sim_shadows = further_after - further_before

# name_data = "le_petit_prince"

# nb_min_ses = 0
# nb_max_ses = 2
#
# nb_min_task = 0
# nb_max_task = 4
###


class PATHS:
    path_file = Path("./data_path.txt")
    if not path_file.exists():
        data = Path(input("data_path?"))
        assert data.exists()
        with open(path_file, "w") as f:
            f.write(str(data) + "\n")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))
    assert data.exists()
    bids = data / name_data


def segment(raw, nb_min_task, nb_max_task, regression_phonemes = False):
    ph_info = pd.read_csv(
        "phoneme_info.csv"
    )  # phonation: "v", "uv", what do these mean ? (voiced ? as in ~ vowel)
    list_freqs = []
    # preproc annotations
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0
    meta["is_session_0"] = False
    meta["is_session_1"] = False
    for task in range(nb_min_task, nb_max_task):
        meta["is_task_" + str(task)] = False

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    assert len(phonemes)
    # d_phonemes = {}
    # d_phonemes["phonation"] = []
    # d_phonemes["manner"] = []
    # d_phonemes["place"] = []
    # d_phonemes["frontback"] = []
    # d_phonemes["roundness"] = []
    # d_phonemes["centrality"] = []
    # all_elements_phonemes = []
    if regression_phonemes == True:
        all_elements_phonemes = [
            ["v", "v", "lo", "b", "u", "c"],
            ["v", "v", "lo", "f", "u", "f"],
            ["v", "v", "m", "b", "r", "f"],
            ["v", "v", "m", "f", "u", "f"],
            ["v", "o", "l", "f", "u", "n"],
            ["uv", "f", "c", "f", "u", "n"],
            ["v", "o", "c", "f", "u", "n"],
            ["v", "f", "d", "f", "u", "n"],
            ["v", "a", "v", "b", "u", "f"],
            ["v", "v", "h", "f", "u", "f"],
            ["uv", "f", "d", "f", "u", "n"],
            ["v", "o", "v", "b", "u", "n"],
            ["uv", "f", "g", "b", "u", "n"],
            ["v", "f", "c", "f", "u", "n"],
            ["uv", "o", "v", "b", "u", "n"],
            ["v", "a", "c", "f", "u", "n"],
            ["v", "n", "l", "f", "u", "n"],
            ["v", "n", "c", "f", "u", "n"],
            ["v", "n", "v", "b", "u", "n"],
            ["v", "v", "h", "b", "r", "f"],
            ["uv", "o", "l", "f", "u", "n"],
            ["v", "a", "v", "b", "u", "n"],
            ["uv", "o", "c", "f", "u", "n"],
            ["v", "a", "l", "f", "r", "n"],
        ]
        d_phonemes = {
            "phonation": ["v", "uv"],
            "manner": ["v", "o", "f", "a", "n"], # how the sound stops
            "place": ["lo", "m", "l", "c", "d", "v", "h", "g"],
            "frontback": ["b", "f"],
            "roundness": ["u", "r"],
            "centrality": ["c", "f", "n"],
        }

    names = []
    for ph, d in phonemes.groupby("phoneme"):
        meta.loc[d.index, "name"] = 0
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"
        # list_els_ph = []
        # for (n, key_item), value in zip(enumerate(match.iloc[0].keys()), d_phonemes.values()):
        #     element= match.iloc[0][n+1]
        #     if element not in value:
        #         value.append(element)
        #     list_els_ph.append(element)
        # if list_els_ph not in all_elements_phonemes:
        #     all_elements_phonemes.append(list_els_ph)
        # print("all_elements_phonemes =", all_elements_phonemes)
        # phonemes = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
        if regression_phonemes == True:
            for num_cat, el_category in enumerate(all_elements_phonemes):
                phonation = el_category[0]
                manner = el_category[1]
                place = el_category[2]
                frontback = el_category[3]
                roundness = el_category[4]
                centrality = el_category[5]
                name = (
                    phonation
                    + "_"
                    + manner
                    + "_"
                    + place
                    + "_"
                    + frontback
                    + "_"
                    + roundness
                    + "_"
                    + centrality
                )
                bool_name = (
                    match.iloc[0].phonation == phonation
                    and match.iloc[0].manner == manner
                    and match.iloc[0].place == place
                    and match.iloc[0].frontback == frontback
                    and match.iloc[0].roundness == roundness
                    and match.iloc[0].centrality == centrality
                )
                meta.loc[d.index, name] = bool_name
                if bool_name == True:
                    meta.loc[d.index, "name"] = num_cat+1
                if name not in names:
                    names.append(name)
            # print(meta.loc[d.index, name])

    # compute word frequency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10  # why ?
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values
    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2
    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))  # what is sfreq ?
    ].astype(int)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim= 40, #20, # 20 #10 #matrices de 20x20 pixels
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )

    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline((-0.2, 0.0))
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline((-0.2, 0.0))
    # print("list_freqs =",list_freqs)
    return epochs

def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    if (SX2 * SY2).any() != 0:
        result = SXY / (SX2 * SY2)
    else:
        result = 0
    return result

def plot(result):
    fig, ax = plt.subplots(1, figsize=[6, 6])
    sns.lineplot(x="time", y="AUC", data=result, hue="label", ax=ax)
    ax.axhline(0, color="k")
    return fig

def _get_epochs(subject, nb_min_task, nb_max_task, nb_min_ses, nb_max_ses):
    all_epochs = list()
    for session in range(nb_min_ses, nb_max_ses):
        print("session ", session)

        for task in range(nb_min_task, nb_max_task):
            print("task ", task)
            print(".", end="")
            bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=str(session),
                task=str(task),
                datatype="meg",
                root=PATHS.bids,
            )
            try:
                raw = mne_bids.read_raw_bids(bids_path)
            except FileNotFoundError:
                print("missing", subject, session, task)
                continue
            raw = raw.pick_types(
                meg=True, stim=False, misc=False, eeg=False, eog=False, ecg=False
            )

            raw.load_data().filter(0.5, 30.0, n_jobs=1)
            epochs = segment(raw, nb_min_task, nb_max_task)
            epochs.metadata["half"] = np.round(np.linspace(0, 1.0, len(epochs))).astype(
                int
            )
            epochs.metadata["task"] = task
            epochs.metadata["session"] = session
            # epochs.metadata["is_session_0"] = True
            epochs.metadata["is_session_" + str(session)] = True
            epochs.metadata["is_task_" + str(task)] = True
            all_epochs.append(epochs)

    if not len(all_epochs):
        return
    epochs = mne.concatenate_epochs(all_epochs)
    m = epochs.metadata
    label = (
        "t"
        + m.task.astype(str)
        + "_s"
        + m.session.astype(str)
        + "_h"
        + m.half.astype(str)
    )
    epochs.metadata["label"] = label
    return epochs


def _decod_one_subject(
    report_TG, subject, target, epochs, nb_min_task, nb_max_task, nb_min_ses, nb_max_ses, model0, score_fct, bool_several_shadow = True):
    #epochs = _get_epochs(subject, nb_min_task, nb_max_task, nb_min_ses, nb_max_ses)
    if epochs is None:
        return
    if target == "words":
        words = epochs["is_word"]
        groups = words.metadata.sequence_id
        print("groups =", groups)
        #for session in range(nb_min_ses, nb_max_ses):#
        #print("session = ", session, target)#
        #words_ses = words0["is_session_" + str(session)] #
        #words = words_ses
        X_words = words.get_data() * 1e13
        evo = words.average()
        fig_evo = evo.plot(spatial_colors=True, show=False)
        y_words = words.metadata.shift(1).wordfreq.values
        decod_specific_label(
            groups,
            report_TG,
            words.times,
            "subject_" + str(subject), #+ "_session_" + str(session), #
            y_words,
            X_words,
            model0,
            score_fct,
            target,
            bool_several_shadow
        )
        return fig_evo

    if target == "vowels":
        phonemes = epochs["not is_word"]
        groups = phonemes.metadata.sequence_id
        print("groups =", groups)
        evo_ph = phonemes.average()
        fig_evo_ph = evo_ph.plot(spatial_colors=True, show=False)
        X_ph = phonemes.get_data() * 1e13
        y_ph = phonemes.metadata["voiced"].values
        # X_freq = phonemes.metadata["frequency"].values
        decod_specific_label(
            groups,
            report_TG,
            phonemes.times,
            "subject_"
            + str(subject),
            #+ "_session_"
            #+ str(session),  # +"_task_"+ str(task),
            y_ph,
            X_ph,
            model0,
            score_fct,
            target,
            bool_several_shadow
        )
        return fig_evo_ph

    if target == "phonemes":
        phonemes = epochs["not is_word"]
        groups = phonemes.metadata.sequence_id
        print("groups =", groups)
        #for session in range(nb_min_ses, nb_max_ses):
        #phonemes = phonemes0["is_session_" + str(session)]
        evo_ph = phonemes.average()
        fig_evo_ph = evo_ph.plot(spatial_colors=True, show=False)
        X_ph = phonemes.get_data() * 1e13
        y_ph = phonemes.metadata["name"].values
        for n_shift in range(-4, 5) :
            print("y_ph =", n_shift, phonemes.metadata.shift(n_shift).name.values)
        # X_freq = phonemes.metadata["frequency"].values
        decod_specific_label(
            groups,
            report_TG,
            phonemes.times,
            "subject_"
            + str(subject),
            #+ "_session_"
            #+ str(session),  # +"_task_"+ str(task),
            y_ph,
            X_ph,
            model0,
            score_fct,
            target,
            bool_several_shadow
        )
        return fig_evo_ph



def decod_specific_label(groups, report_TG, times, label, y, X, model0, score_fct, word_phoneme, bool_several_shadow = True, tg_bool=True):
    if bool_several_shadow == True :
        scores_shift_mean = decod(groups, X, y, word_phoneme, label, times, model0=model0, score_fct=score_fct)
        if tg_bool == True:
            for n_shift, shift in enumerate(range(further_before, further_after)):  # (-4, 5)
                label_shift = label + " - shift = " +str(shift)
                print("decod_specific_label scores_shift_mean ", scores_shift_mean)
                score_matrix = scores_shift_mean[n_shift, :, :]
                print("score_matrix =", shift, score_matrix )
                # score_matrix in score_matrices:
                fig_each_ses_task_subject = plot_TG_matrix(
                    times, score_matrix, score_fct
                )
                report_TG.add_figure(fig_each_ses_task_subject, label_shift, tags=word_phoneme)
                report_TG.save("decoding_TG.html", open_browser=False, overwrite=True)
        return score_matrix
    else:
        score_matrix = decod(groups, X, y, word_phoneme, label, times, model0=model0, score_fct=score_fct,
                             bool_several_shadow = bool_several_shadow)
        if tg_bool == True:
            fig_each_ses_task_subject = plot_TG_matrix(
                times, score_matrix, score_fct
            )
            report_TG.add_figure(fig_each_ses_task_subject, label, tags=word_phoneme)
            report_TG.save("decoding_TG.html", open_browser=False, overwrite=True)
        return score_matrix

def decod(groups, X, y, word_phoneme, label, times, model0, score_fct, bool_several_shadow = True):
    assert len(X) == len(y)
    if word_phoneme == "words":
        y = scale(y[:, None])[:, 0]
        if len(set(y[:1000])) > 2:
            y = y > np.nanmedian(y)
    if bool_several_shadow == True :
        score_means = cross_val_score(
            groups, X, y, label, times, model0=model0, score_fct=score_fct)
        print("decod score_means =", score_means)
        return score_means
    else:
        score_mean = cross_val_score(
            groups, X, y, label, times, model0=model0, score_fct=score_fct, bool_several_shadow = bool_several_shadow)
        print("score_mean =", score_mean)
        return score_mean


def shift_homade(arr, num, fill_value='nan'): #np.nan ValueError: cannot convert float NaN to integer
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def shift_y(y, shift):
    if shift > 0 :
        y_ = y[shift:]

    elif shift < 0 :
        y_ = y[:shift]
    else :
        y_ = y
    return y_

def shift_X(X, shift):
    if shift > 0:
        X_ = X[:-shift]

    elif shift < 0:
        X_ = X[-shift:]
    else:
        X_ = X
    return X_


def modify_y_preds(y_preds, num):
    if num > 0 :
        y_preds_new = y_preds[num]

def cross_val_score(groups, X, y, label, times, model0=LinearDiscriminantAnalysis(), score_fct=roc_auc_score,
                    bool_several_shadow = True, score_t1_bool=True, tg_bool=True):
    model = make_pipeline(StandardScaler(), model0) #LogisticRegression()
    # fit predict
    n, nchans, ntimes = X.shape
    print("X.shape ", X.shape)
    # 27 English speakers who listened to 2 sessions of 1h of naturalistic stories
    # words: n, nchans, ntimes = 668 208 41 (n: nb of trials: words or phonemes)
    # phonemes: n, nchans, ntimes = 1794 208 41 -> test = 1794/15 = 120
    nsplits = 15
    score_mean = 0.5
    cv = KFold(n_splits=nsplits, shuffle=False) # shuffle False ?
    # si shuffle = False: comment on sait si l'on n'est pas en train de de prédire le phonème entendu, qui a des chances d'être associé au phonème d'après on d'avant ?
    # peut-être pcq on voit que le phonème précédent et suivant ne sont pas prédits avec la même précision et la même allure de TG matrix

    scores_along_word = []
    scores = np.zeros((nsplits, ntimes, ntimes))
    scores_shift = np.zeros((nsplits, n_sim_shadows, ntimes, ntimes))
    score_t1_cv = np.zeros((nsplits, ntimes))
    #for split, (train, test) in enumerate(cv.split(X)):
    if bool_several_shadow == True:
        for split, (train, test) in enumerate(cv.split(X, groups=groups)):
            # shadowing on several phonemes
            for n_shift, shift in enumerate(range(further_before, further_after)) :  # (-4, 5)
                # y_ = shift_homade(y, shift, fill_value='nan')
                for t1 in trange(ntimes):
                    model.fit(
                        shift_X(X[train, :, t1], shift), shift_y(y[train].astype(float), shift)
                    )
                    # print("X[train, :, t1] ", X[train, :, t1])
                    # print("shift_X(X[train, :, t1], shift)", shift, shift_X(X[train, :, t1], shift))
                    #
                    # print("y[train].astype(float) ", y[train].astype(float))
                    # print("shift_y(y[train].astype(float), shift)", shift, shift_y(y[train].astype(float), shift))

                    for t2 in trange(ntimes):
                        if tg_bool == True :
                            if score_fct is roc_auc_score :
                                decision = model.decision_function(shift_X(X[test, :, t2], shift))
                                y_preds = np.exp(decision) / np.sum(np.exp(decision))
                                # y_preds = model.predict_proba(X[test, :, t1])[:, 1] (for LDA classifier)
                            else :
                                y_preds = model.predict(shift_X(X[test, :, t2], shift))
                            score = score_fct(shift_y(y[test].astype(float), shift), y_preds)
                            scores_shift[split, n_shift, t1, t2] = score
    else:
        for split, (train, test) in enumerate(cv.split(X, groups=groups)):
            for t1 in trange(ntimes):
                model.fit(
                    X[train, :, t1], y[train].astype(float)
                )
                if score_t1_bool == True:
                    if score_fct is roc_auc_score: # classifier used
                        decision = model.decision_function(X[test, :, t1])
                        y_preds_t1 = np.exp(decision) / np.sum(np.exp(decision))
                        # y_preds_t1 = model.predict_proba(X[test, :, t1])[:, 1]
                    else: # regressor used
                        y_preds_t1 = model.predict(X[test, :, t1])
                    score_t1 = score_fct(y[test].astype(float), y_preds_t1)
                    score_t1_cv[split, t1] = score_t1

                for t2 in trange(ntimes):
                    if tg_bool == True:
                        if score_fct is roc_auc_score :
                            decision = model.decision_function(X[test, :, t2])
                            y_preds = np.exp(decision) / np.sum(np.exp(decision))
                            #y_preds = model.predict_proba(X[test, :, t1])[:, 1] (for LDA classifier)
                        else :
                            y_preds = model.predict(X[test, :, t2])
                        score = score_fct(y[test].astype(float), y_preds)  # best for classification (not for regression)
                        scores[split, t1, t2] = score
    #if score_t1_bool == True:
        #score_t1_mean = score_t1_cv.mean(0)
    # return score_t1_mean
    if bool_several_shadow == True:
        print("scores_shift =", scores_shift[0].shape, scores_shift[0])
        scores_shift_mean = np.nanmean(scores_shift, 0)
        print("scores_shift_mean =", scores_shift_mean.shape, scores_shift_mean)
        return scores_shift_mean
    else:
        scores_mean = scores.mean(0)
        return scores_mean
    # on s'attend à ce que le score de decodabilité augmente une fois le mot/phonème entendu


def plot_TG_matrix(
    times,
    score_matrix,
    score_fct
):
    fig, ax = plt.subplots(1, 1)
    if score_fct is roc_auc_score:
        vmin_, vmax_ = 0.40, 0.60
    else:
        vmin_, vmax_ = -0.10, 0.10
    im = ax.imshow(
        score_matrix,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_,
        vmax=vmax_,
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    plt.xticks(
        [n for n, i in enumerate(times) if n % 5 == 0],
        [i for n, i in enumerate(times) if n % 5 == 0],
    )
    plt.yticks(
        [n for n, i in enumerate(times) if n % 5 == 0],
        [i for n, i in enumerate(times) if n % 5 == 0],
    )
    cbar = plt.colorbar(im, ax=ax)
    if score_fct is roc_auc_score:
        cbar.set_label("ROC AUC")
    else:
        cbar.set_label("R score")
    # cbar.set_label("R score")
    time_inst = time.time()
    # plt.savefig(root + "_" + label + "_" + str(time_inst)[-4 :] + ".png")
    return fig


"""
if __name__ == "__main__" :
    report = mne.Report()
    report_TG = mne.Report()

    ph_info = pd.read_csv("phoneme_info.csv")  # phonation: "v", "uv", what do these mean ? (voiced ? as in ~ vowel)
    subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x : x.split("-")[1]).values
    # decoding
    targets = ["words", "vowels"]
    for target in targets:
        for subject in subjects:
            print(subject)
            out = _decod_one_subject(subject, target)
            if out is None :
                continue
            (
                fig_evo,
                fig_evo_ph,
            ) = out

            report_TG.add_figure(fig_evo, subject, tags="evo_word")
            report_TG.add_figure(fig_evo_ph, subject, tags="evo_phoneme")
            report.save("decoding.html", open_browser=False, overwrite=True)
            report_TG.save("decoding_TG.html", open_browser=False, overwrite=True)
            print("done")
"""
