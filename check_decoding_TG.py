import mne
import mne_bids
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

    bids = data / "bids_anonym"


def segment(raw):
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

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frequency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10 # why ?
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values

    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2

    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2)) # what is sfreq ?
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim=20, #10
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )

    # threshold # what is this threshold for ?
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs


def decod(X, y, meta, times, word_phoneme):
    assert len(X) == len(y) == len(meta) # what is in meta ?
    meta = meta.reset_index()
    if word_phoneme == "words":
        y = scale(y[:, None])[:, 0]
        # print("y is ", len(set(y[:1000])), len(y), set(y[:1000]), y)
        # words: len(set(y[:1000])), len(y) -> 229 668, y.shape (668,) / and phonemes: 2 1794
        if len(set(y[:1000])) > 2: # what is this for ?
            y = y > np.nanmedian(y)
        #else: # ADDED TO TRY IT OUT
            # y = y > np.nanmedian(y) # ADDED TO TRY IT OUT
    print(word_phoneme, " y is ", y)

    # define data
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    model_TG = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, nchans, ntimes = X.shape # what is n and ntimes ?
    # 27 English speakers who listened to 2 sessions of 1h of naturalistic stories
    # words: n, nchans, ntimes = 668 208 41 (n: nb of words ?)
    # phonemes: n, nchans, ntimes = 1794 208 41
    print("n, nchans, ntimes =", n, nchans, ntimes)
    preds = np.zeros((n, ntimes))
    preds_TG = np.zeros((n, ntimes, ntimes))
    scores_TG = []
    for t1 in trange(ntimes):
        preds[:, t1] = cross_val_predict(
           model, X[:, :, t1], y, cv=cv, method="predict_proba" # why predict_proba and not predict ?
        )[:, 1]
        if y[0] == False or y[0] == True:
            y_scal = [int(bool_val == True) for bool_val in y]
        else:
            y_scal = y
            print(y[0], type(y[0]), "scal")
        model_TG.fit(X[:, :, t1], y_scal)
        for t2 in trange(ntimes):
            preds_TG[:, t1, t2] = model_TG.predict(X[:, :, t2]) ####  #[:, 1]
        for label, m in meta.groupby("label"):
            Rs_TG = correlate(y[m.index, None], preds_TG[m.index, t1])
            for t2, r_TG in zip(times, Rs_TG) :
                scores_TG.append(dict(score=r_TG, time_1=t1, time_2=t2, label=label, n=len(m.index))) ## t1 et t2 !
    df_scores_TG = pd.DataFrame(scores_TG)
    print("df_scores_TG", df_scores_TG)
    # score
    out = list()
    for label, m in meta.groupby("label"):
        Rs = correlate(y[m.index, None], preds[m.index])
        for t, r in zip(times, Rs):
            out.append(dict(score=r, time=t, label=label, n=len(m.index)))

    # plot figure
    for label, m in meta.groupby("label"):
        df_scores_TG_reduced = df_scores_TG[df_scores_TG["label"].isin([label])] # dataframe[dataframe['Stream'].isin(options)]
        eg_matrix = np.zeros((ntimes, ntimes))
        t2_int_list = [t2_int for t2_int in trange(ntimes)]
        for t1 in trange(ntimes):
            for t2, t2_int in zip(times, t2_int_list):
                df_scores_TG_reduced_t1 = df_scores_TG_reduced[df_scores_TG_reduced["time_1"] == t1]
                df_scores_TG_reduced_t2 = df_scores_TG_reduced_t1[df_scores_TG_reduced_t1["time_2"] == t2]
                eg_matrix[t1, t2_int] = df_scores_TG_reduced_t2.iloc[:]["score"].astype(float)
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(eg_matrix, interpolation='lanczos', origin='lower', cmap='RdBu_r', vmin=0., vmax=0.5)
        ax.set_xlabel('Testing Time (s)')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Temporal generalization')
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        plt.xticks([i for n,i in enumerate(t2_int_list) if n%5 == 0], [i for n,i in enumerate(times) if n%5 == 0])
        plt.yticks([i for n,i in enumerate(t2_int_list) if n%5 == 0], [i for n,i in enumerate(times) if n%5 == 0])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('AUC')
        time_inst = time.time()
        plt.savefig("/Users/Josephine/Desktop/tg_figures/figure_plot_" + word_phoneme + "_" + label + "_" + str(time_inst)[-4:]+".png")
        report_TG.add_figure(fig, label, tags= word_phoneme)

    return pd.DataFrame(out)


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
        print("denom ", SX2 * SY2)
        result = SXY / (SX2 * SY2)
    else:
        result = 0
    return result


def plot(result):
    fig, ax = plt.subplots(1, figsize=[6, 6])
    sns.lineplot(x="time", y="score", data=result, hue="label", ax=ax)
    ax.axhline(0, color="k")
    return fig


ph_info = pd.read_csv("phoneme_info.csv") # phonation: "v", "uv", what do these mean ?
subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values

nb_max_ses = 2
nb_min_task = 0
nb_max_task = 1
def _get_epochs(subject):
    all_epochs = list()
    for session in range(nb_max_ses):
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
            epochs = segment(raw)
            epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(epochs))
            ).astype(int)
            epochs.metadata["task"] = task
            epochs.metadata["session"] = session

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


def _decod_one_subject(subject):
    epochs = _get_epochs(subject)
    if epochs is None:
        return
    # words
    print("# words")
    words = epochs["is_word"]
    print("words = epochs[is_word]")
    evo = words.average()
    print("evo", evo)
    fig_evo = evo.plot(spatial_colors=True, show=False)
    X = words.get_data() * 1e13
    y = words.metadata["wordfreq"].values
    print("X,y")
    results = decod(X, y, words.metadata, words.times, "words")
    results["subject"] = subject
    results["contrast"] = "wordfreq"
    print("results")
    fig_decod = plot(results)
    print("fig_decod")

    # Phonemes
    print("# Phonemes")
    phonemes = epochs["not is_word"]
    print("phonemes = epochs[not is_word]")
    evo = phonemes.average()
    print("evo", evo)
    fig_evo_ph = evo.plot(spatial_colors=True, show=False)
    X = phonemes.get_data() * 1e13
    y = phonemes.metadata["voiced"].values
    print("phonemes.metadata[voiced] ", phonemes.metadata["voiced"])
    print("y phonemes ", y)
    results_ph = decod(X, y, phonemes.metadata, phonemes.times, "phonemes")
    results_ph["subject"] = subject
    results_ph["contrast"] = "voiced"
    print("results_ph")
    fig_decod_ph = plot(results_ph)
    print("fig_decod_ph")
    return fig_evo, fig_decod, results, fig_evo_ph, fig_decod_ph, results_ph


if __name__ == "__main__":
    report = mne.Report()
    report_TG = mne.Report()

    # decoding
    all_results = list()
    results = list()
    for subject in subjects:
        print(subject)
        out = _decod_one_subject(subject)
        if out is None:
            continue

        (
            fig_evo,
            fig_decod,
            results,
            fig_evo_ph,
            fig_decod_ph,
            results_ph,
        ) = out

        report.add_figure(fig_evo, subject, tags="evo_word")
        report.add_figure(fig_decod, subject, tags="word")
        report.add_figure(fig_evo_ph, subject, tags="evo_phoneme")
        report.add_figure(fig_decod_ph, subject, tags="phoneme")
        report.save("decoding.html", open_browser=False, overwrite=True)

        report_TG.save("decoding_TG.html", open_browser=False, overwrite=True)

        all_results.append(results)
        all_results.append(results_ph)
        print("done")

    pd.concat(all_results, ignore_index=True).to_csv("decoding_results.csv")
