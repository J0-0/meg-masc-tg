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
matplotlib.use("Agg")
mne.set_log_level(False)

### constants ###
nb_min_ses = 0
nb_max_ses = 2

nb_min_task = 0
nb_max_task = 1
###

class PATHS :
    path_file = Path("./data_path.txt")
    if not path_file.exists() :
        data = Path(input("data_path?"))
        assert data.exists()
        with open(path_file, "w") as f :
            f.write(str(data) + "\n")
    with open(path_file, "r") as f :
        data = Path(f.readlines()[0].strip("\n"))
    assert data.exists()
    bids = data / "bids_anonym"

def segment(raw) :
    # preproc annotations
    meta = list()
    for annot in raw.annotations :
        d = eval(annot.pop("description"))
        for k, v in annot.items() :
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0
    meta["is_session_0"] = False
    meta["is_session_1"] = False
    for task in range(nb_min_task, nb_max_task):
        meta["is_task_"+str(task)] = False

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    #print("phonemes ! ", [(ph, len(d.index), d) for (ph, d) in phonemes.groupby("phoneme")])
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme") :
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"
        # meta.loc[d.index, "frequency"] = np.round(len(d.index)/len(phonemes), 2) then d[frequency] = [auc_score1, auc_score2, etc.] et plor auc average as a fct of frequency

    # compute word frequency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10  # why ?
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x : zipf_frequency(x, "en")  # noqa
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
        decim=20,  # 10
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
    return epochs

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

def plot(result):
    fig, ax = plt.subplots(1, figsize=[6, 6])
    sns.lineplot(x="time", y="AUC", data=result, hue="label", ax=ax)
    ax.axhline(0, color="k")
    return fig

def _get_epochs(subject):
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
            try :
                raw = mne_bids.read_raw_bids(bids_path)
            except FileNotFoundError:
                print("missing", subject, session, task)
                continue
            raw = raw.pick_types(
                meg=True, stim=False, misc=False, eeg=False, eog=False, ecg=False
            )

            raw.load_data().filter(0.5, 30.0, n_jobs=1)
            epochs = segment(raw)
            epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(epochs))
            ).astype(int)
            epochs.metadata["task"] = task
            epochs.metadata["session"] = session
            #epochs.metadata["is_session_0"] = True
            epochs.metadata["is_session_" + str(session)] = True
            epochs.metadata["is_task_" + str(task)] = True
            all_epochs.append(epochs)

    if not len(all_epochs) :
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

def _decod_one_subject(subject) :
    epochs = _get_epochs(subject)
    if epochs is None:
        return
    # words
    words0 = epochs["is_word"]
    for session in range(nb_min_ses, nb_max_ses):
        print("session = ", session)
        words_ses = words0["is_session_"+ str(session)]
        #for task in range(nb_min_task, nb_max_task) :
        #   print("task = ", task)
        #   words = words_ses["is_task_" + str(task)]
        words = words_ses
        X_words = words.get_data() * 1e13
        y_words = words.metadata["wordfreq"].values
        #print("X_words, y_words ", X_words.shape, y_words.shape)
        #print("X_words =", X_words)
        #print("y_words =", y_words)
        #print("words =", words)
        evo = words.average()
        fig_evo = evo.plot(spatial_colors=True, show=False)
        #print("session ", session, [X_words[i, 0, 0] for i in range(9)])
        decod_specific_label(words.times,
                             "subject_"+ str(subject) +"_session_"+ str(session),
                             y_words, X_words, word_phoneme = "words")

    # Phonemes
    phonemes0 = epochs["not is_word"]
    print("phonemes0 =", phonemes0)
    for session in range(nb_min_ses, nb_max_ses):
        print("session = ", session)
        phonemes_ses = phonemes0["is_session_" + str(session)]
        #for task in range(nb_min_task, nb_max_task) :
        #    print("task = ", task)
        #    phonemes = phonemes_ses["is_task_" + str(task)]
        phonemes = phonemes_ses
        evo_ph = phonemes.average()
        fig_evo_ph = evo_ph.plot(spatial_colors=True, show=False)
        X_ph = phonemes.get_data() * 1e13
        y_ph = phonemes.metadata["voiced"].values
        #print("X_ph, y_ph ", X_ph.shape, y_ph.shape)
        #print("phonemes =", phonemes)
        #print("X_ph =", X_words)
        #print("y_ph =", y_words)
        decod_specific_label(phonemes.times,
                             "subject_"+ str(subject) +"_session_"+ str(session), #+"_task_"+ str(task),
                             y_ph, X_ph, word_phoneme = "phonemes")
    return fig_evo, fig_evo_ph

def decod_specific_label(times, label, y, X, word_phoneme, tg_bool=True):
    score_matrix = decod(X, y, word_phoneme, label, times)
    if tg_bool == True :
        fig_each_ses_task_subject = plot_TG_matrix(times, label, word_phoneme, score_matrix)
        report_TG.add_figure(fig_each_ses_task_subject, label, tags=word_phoneme)
        report_TG.save("decoding_TG.html", open_browser=False, overwrite=True)

def decod(X, y, word_phoneme, label, times):
    assert len(X) == len(y)
    if word_phoneme == "words": #or word_phoneme == "phonemes":
        y = scale(y[:, None])[:, 0]
        if len(set(y[:1000])) > 2:
            y = y > np.nanmedian(y)
    #print(word_phoneme, y)
    score_mean = cross_val_score(X, y, label, word_phoneme, times)  # for each session, subject, h0/h1, etc. (m.label, roc_auc)
    return score_mean

def cross_val_score(X,y_0, label, word_phoneme, times, score_t1_bool = True, tg_bool = True):  # for each session, subject, h0/h1, etc. (m.label, roc_auc)
    model = make_pipeline(StandardScaler(),
                          LinearDiscriminantAnalysis())
    #model = make_pipeline(StandardScaler(), LogisticRegression())

    # fit predict
    n, nchans, ntimes = X.shape
    #print("X.shape", X.shape)
    y = y_0[:] # is it useful ?
    # 27 English speakers who listened to 2 sessions of 1h of naturalistic stories
    # words: n, nchans, ntimes = 668 208 41 (n: nb of trials: words or phonemes)
    # phonemes: n, nchans, ntimes = 1794 208 41
    nsplits = 15
    score_mean = 0.5
    cv = KFold(n_splits=nsplits, shuffle=True, random_state=0)
    scores = np.zeros((nsplits, ntimes, ntimes))
    score_t1_cv = np.zeros((nsplits, ntimes))
    for split, (train, test) in enumerate(cv.split(X)):
        for t1 in trange(ntimes):
            model.fit(X[train, :, t1], y[train].astype(float))  # n_trial -> chaque phoneme/word
            # is it better to mix the training and testing data for this dataset or not ?
            # yes for homogeneisation but careful for independance too
            # solut°: cv qui prend des segments d'une dizaine de secondes pour ne pas avoir d'élements trop proches
            if score_t1_bool == True:
                y_preds_t1 = model.predict_proba(X[test, :, t1])[:, 1]
                score_t1 = sklearn.metrics.roc_auc_score(y[test].astype(float), y_preds_t1)
                score_t1_cv[split, t1] = score_t1
            for t2 in trange(ntimes):
                if tg_bool == True:
                    #y_preds0 = model.decision_function(X[test, :, t2])
                    #y_preds = np.exp(y_preds0) / np.sum(np.exp(y_preds0))
                    y_preds = model.predict_proba(X[test, :, t2])[:, 1]
                    #score = correlate(y[test].astype(float), y_preds)
                    score = sklearn.metrics.roc_auc_score(y[test].astype(float), y_preds)  # best for classification (not for regression)
                    scores[split, t1, t2] = score
                    score_mean = scores.mean(0)
    if score_t1_bool == True:
        score_t1_mean = score_t1_cv.mean(0)
        df_results = pd.DataFrame()
        df_results["time"] = times
        df_results["AUC"] = score_t1_mean
        df_results["label"] = label
        fig_decod = plot(df_results)
        report_TG.add_figure(fig_decod, label, tags=word_phoneme)
        #report_TG.save("decoding_TG.html", open_browser=True, overwrite=True)
    return score_mean
    # on s'attend à ce que le score de decodabilité augmente une fois le mot/phonème entendu

def plot_TG_matrix(times, label, word_phoneme, score_matrix, root = "/Users/Josephine/Desktop/tg_figures/figure_plot_"):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(score_matrix, interpolation='lanczos', origin='lower', cmap='RdBu_r', vmin=0.35, vmax=0.65)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal generalization')
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.xticks([n for n, i in enumerate(times) if n % 5 == 0], [i for n, i in enumerate(times) if n % 5 == 0])
    plt.yticks([n for n, i in enumerate(times) if n % 5 == 0], [i for n, i in enumerate(times) if n % 5 == 0])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ROC AUC')
    #cbar.set_label("R score")
    time_inst = time.time()
    #plt.savefig(root + word_phoneme + "_" + label + "_" + str(time_inst)[-4 :] + ".png")
    return fig


if __name__ == "__main__" :
    report = mne.Report()
    report_TG = mne.Report()

    ph_info = pd.read_csv("phoneme_info.csv")  # phonation: "v", "uv", what do these mean ? (voiced ? as in ~ vowel)
    subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x : x.split("-")[1]).values
    # decoding
    for subject in subjects:
        print(subject)
        out = _decod_one_subject(subject)
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















