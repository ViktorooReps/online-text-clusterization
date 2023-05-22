import json
import os
import pickle
from typing import Optional

import requests
from pymystem3 import Mystem

url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"


def get_text(url, encoding='utf-8', to_lower=True):
    url = str(url)
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception('parameter [url] can be either URL or a filename')


def get_stopwords():
    return get_lemmatizer()(get_text(url_stopwords_ru).replace('\n', ' '))


def load_dev():
    with open('data/dev-dataset-task2022-04_preprocessed.json') as f:
        json_dataset = json.load(f)
    texts, labels = zip(*json_dataset)

    return texts, labels


def save(model, path: str):
    pickle.dump(model, open(path, 'wb'))


def load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_lemmatizer(limit_length=0):
    lemmatizer = Mystem().lemmatize

    def limit_length_wrapper(text):
        lemmatized = lemmatizer(text)
        lemmatized = filter(lambda x: len(x) > limit_length, lemmatized)
        return list(lemmatized)

    return limit_length_wrapper


def postprocess(labels, last_label: Optional[int] = None, outlier_label: int = -1):
    if last_label is None:
        last_label = max(labels)

    def unique_label():
        nonlocal last_label
        last_label += 1
        return last_label

    return [label if label != outlier_label else unique_label() for label in labels]


if __name__ == '__main__':
    save(get_stopwords(), 'stopwords.pkl')
