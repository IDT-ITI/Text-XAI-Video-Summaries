import os
import warnings
import pandas as pd
import numpy as np

EPSILON = 1e-10

# Paths to the main folders
summe_folder = '../data/SumMe'
tvsum_folder = '../data/TVSum'

# Excluded indices per videoset
videosets = {
    "VideoSet1": {
        "SumMe": {1, 3, 5, 11, 13, 16, 20, 24, 25},
        "TVSum": {3, 15, 21}
    },
    "VideoSet2": {
        "SumMe": {1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25},
        "TVSum": {1, 3, 6, 12, 15, 21, 29, 37, 39, 41, 46}
    }
}

# All available indices
dataset = ["SumMe", "TVSum"]
all_video_indices = {
    "SumMe": list(range(1, 26)),
    "TVSum": list(range(1, 51))
}

def fragments_explanation_scores(video_path, fragment_scores_1, fragment_scores_2, fragment_scores_3):
    """ Computes the fragment-level explanation scores."""
    df = pd.read_csv(os.path.join(video_path, "fragments_explanation_evaluation_metrics.csv"))
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    video_scores = df.to_numpy()

    video_scores[video_scores == 1.0] = 1.0 - EPSILON
    video_scores[video_scores == -1.0] = -1.0 + EPSILON

    if not np.isnan(video_scores[0][0]) and not np.isnan(video_scores[0][1]):
        temp = np.copy(video_scores[:1])
        temp[:, 0:4] = np.arctanh(temp[:, 0:4])
        fragment_scores_1.append(temp)

    if not np.isnan(video_scores[1][0]) and not np.isnan(video_scores[1][1]):
        temp = np.copy(video_scores[:2])
        temp[:, 0:4] = np.arctanh(temp[:, 0:4])
        fragment_scores_2.append(temp)

    if not np.isnan(video_scores[2][0]) and not np.isnan(video_scores[2][1]):
        temp = np.copy(video_scores[:3])
        temp[:, 0:4] = np.arctanh(temp[:, 0:4])
        fragment_scores_3.append(temp)

    return fragment_scores_1, fragment_scores_2, fragment_scores_3

# Run evaluation for all videosets
for videoset_key, excluded in videosets.items():

    # Build valid video indices after exclusion
    videos = [
        [v for v in all_video_indices["SumMe"] if v not in excluded["SumMe"]],
        [v for v in all_video_indices["TVSum"] if v not in excluded["TVSum"]]
    ]

    for d in range(len(dataset)):
        fragment_scores_1 = []
        fragment_scores_2 = []
        fragment_scores_3 = []

        for vid in videos[d]:
            video_path = f"../data/{dataset[d]}/video_{vid}/visual_explanation/"
            if not os.path.exists(os.path.join(video_path, "fragments_explanation_evaluation_metrics.csv")):
                continue
            fragment_scores_1, fragment_scores_2, fragment_scores_3 = fragments_explanation_scores(
                video_path, fragment_scores_1, fragment_scores_2, fragment_scores_3
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            fragment_scores_1 = np.nanmean(fragment_scores_1, axis=0)
            fragment_scores_1[:, 0:4] = np.tanh(fragment_scores_1[:, 0:4])

            fragment_scores_2 = np.nanmean(fragment_scores_2, axis=0)
            fragment_scores_2[:, 0:4] = np.tanh(fragment_scores_2[:, 0:4])

            fragment_scores_3 = np.nanmean(fragment_scores_3, axis=0)
            fragment_scores_3[:, 0:4] = np.tanh(fragment_scores_3[:, 0:4])

        scores_path = f"./final_scores/{dataset[d]}/{videoset_key}/"
        os.makedirs(scores_path, exist_ok=True)

        columns_names = [
            "Attention Disc Plus", "Lime Disc Plus"
        ]

        df1 = pd.DataFrame(fragment_scores_1, columns=columns_names)
        df1.index = ['Top 1']
        df2 = pd.DataFrame(fragment_scores_2)
        df2.index = ['Top 1', 'Top 2']
        df3 = pd.DataFrame(fragment_scores_3)
        df3.index = ['Top 1', 'Top 2', 'Top 3']

        with open(os.path.join(scores_path, "fragment_explanation_scores.csv"), 'w') as f:
            df1.to_csv(f)
            f.write('\n')
            df2.to_csv(f, header=False)
            f.write('\n')
            df3.to_csv(f, header=False)

