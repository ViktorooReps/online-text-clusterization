import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from transformers.utils import to_numpy

from data.preprocess import is_corrupted
from encoder import Encoder
from metric import bcubed_f1

try:
    from tqdm import tqdm
except ImportError:
    print('Unable to import tqdm!')
    tqdm = lambda x: x


def visualize(features: np.ndarray, labels: np.ndarray):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        num_examples = len(labels)
        print(features.shape, labels.shape)

        pca_features = TSNE(n_components=2, perplexity=min(num_examples - 1, 30), init='random').fit_transform(features)
        sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=labels)
        plt.savefig(f'plots/plot_{num_examples}ex.png')
        plt.clf()
    except Exception:
        print(f'Visualization failed with {traceback.format_exc()}')


def fit_ensemble(x: np.ndarray, y: np.ndarray, *, n_estimators: int) -> VotingClassifier:
    n_estimators = max(min(len(y) // 5, n_estimators), 1)
    last_label = int(max(y))
    outlier_label = last_label + 1

    if n_estimators < 2:
        estimator = RadiusNeighborsClassifier(
            n_jobs=4,
            outlier_label=outlier_label,
            radius=0.55,
        ).fit(x, y)
        named_estimators = [('estimator', estimator)]
        trained_estimators = [estimator]
    else:
        kfold = KFold(n_splits=n_estimators, shuffle=True)

        named_estimators = []
        trained_estimators = []
        for estimator_idx, (train_index, test_index) in enumerate(kfold.split(x)):
            estimator = RadiusNeighborsClassifier(
                n_jobs=4,
                outlier_label=outlier_label,
                radius=0.55,
            ).fit(x[train_index], y[train_index])

            named_estimators.append((f'estimator_{estimator_idx}', estimator))
            trained_estimators.append(estimator)

    ensemble = VotingClassifier(named_estimators)
    ensemble.estimators_ = trained_estimators
    ensemble.le_ = LabelEncoder().fit(list(range(outlier_label + 1)))
    ensemble.classes_ = ensemble.le_.classes_
    return ensemble


class Solution:
    encoder = Encoder.load(Path('main.pkl'))
    estimator = None

    history_feats = []
    history_labels = []

    new_labels = 0

    @classmethod
    def predict(cls, text: str) -> str:
        input_ids = cls.encoder.prepare_inputs([text])['input_ids']
        feats = to_numpy(cls.encoder(input_ids))
        if len(cls.history_labels):
            label = cls.estimator.predict(feats)[0]
        else:
            label = 0

        if is_corrupted(text):
            return str(label)  # do not save into training data

        cls.new_labels += 1

        cls.history_feats.append(feats)
        cls.history_labels.append(label)

        stacked_features = np.concatenate(cls.history_feats, axis=0)
        labels = np.array(cls.history_labels, dtype=int)

        if cls.new_labels >= len(cls.history_labels) * 0:
            cls.estimator = fit_ensemble(stacked_features, labels, n_estimators=5)
            cls.new_labels = 0

        return str(cls.history_labels[-1])

    @classmethod
    def evaluate(cls) -> Dict[str, float]:
        with open('data/dev-dataset-task2022-04.json') as f:
            json_dataset = json.load(f)
        texts, labels = zip(*json_dataset)

        predictions = []
        for text in tqdm(texts):
            predictions.append(cls.predict(text))
            if not len(predictions) % 100:
                print(bcubed_f1(predictions, labels[:len(predictions)]))
        return {'bcubed_f1': bcubed_f1(predictions, labels)}


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

    st_time = time.time()
    print(Solution.evaluate())
    print(f'End: {time.time() - st_time}s')
