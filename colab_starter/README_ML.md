# Colab Starter Kit для ML-команды (SMS Guard, Контракт v3.1 Pickle)

В этой папке лежат **точные копии** модулей ядра, чтобы ваше обучение в Google Colab на 100% совпадало с инференсом в репозитории.

## Что загрузить в Google Colab
Скопируйте в текущую рабочую папку Colab (`/content/`):
- `config.py`
- `preprocessing.py`
- `features.py`
- `scoring.py`
- `export_pkl.py`
- `train_colab_template.py` (готовый 1-click шаблон обучения и экспорта!)
- `sms_train.csv` и `sms_dev.csv` (из `data/`)

Быстрый запуск в Colab (после загрузки файлов): `!python train_colab_template.py`.

## Минимальный пример обучения и экспорта `model.pkl`
```python
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import FEATURE_NAMES
from preprocessing import normalize, strip_urls_for_text_features, unify_kyrgyz
from features import extract_features
from scoring import logit_to_score_and_risk, compute_binary_metrics
from export_pkl import export_sklearn_to_model_pkl, verify_model_pkl

train_df = pd.read_csv("sms_train.csv")
dev_df = pd.read_csv("sms_dev.csv")

def prepare(texts):
    feats, clean_texts = [], []
    for t in texts:
        nm = normalize(t)
        vals, _ = extract_features(nm)
        feats.append([vals[f] for f in FEATURE_NAMES])
        c = strip_urls_for_text_features(nm["clean_text"], nm["domains"])
        tr = strip_urls_for_text_features(nm["translit_text"], nm["domains"])
        clean_texts.append(unify_kyrgyz(f"{c} {tr}".strip()))
    return csr_matrix(feats), clean_texts

X_tr_12, tr_texts = prepare(train_df["text"])
y_tr = (train_df["label"] == "fraud").astype(int).values

word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=250, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=350, sublinear_tf=True)

X_tr = hstack([X_tr_12, word_vec.fit_transform(tr_texts), char_vec.fit_transform(tr_texts)])
clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
clf.fit(X_tr, y_tr)

# Подберите score_shift через 5-fold CV на sms_train.csv под порог score >= 65!
best_shift = 0.6

export_sklearn_to_model_pkl(
    clf=clf,
    out_path="model.pkl",
    word_vectorizer=word_vec,
    char_vectorizer=char_vec,
    score_shift=best_shift,
    train_csv_path="sms_train.csv",
)
verify_model_pkl("model.pkl")
```

Полный регламент, формулы и чек-лист приёмки — в [Контракт/FINAL_CONTRACT_ML_PKL.md](../../Контракт/FINAL_CONTRACT_ML_PKL.md).
