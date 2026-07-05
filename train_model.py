import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from preprocessing import preprocess_data


def train_and_save(data_path='data/yield_df.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    raw_df = pd.read_csv(data_path)
    original_df, df_final, scaler, le = preprocess_data(raw_df)

    X = df_final.drop(columns=['hg/ha_yield', 'Yield_Class'])
    y = le.transform(df_final['Yield_Class'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Decision Tree ────────────────────────────────────────────────────────
    dt_model = DecisionTreeClassifier(
        criterion='gini', splitter='best',
        max_depth=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    # 5-Fold Cross Validation (DT)
    cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')

    # ── Naive Bayes ──────────────────────────────────────────────────────────
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # ── Save ─────────────────────────────────────────────────────────────────
    joblib.dump(dt_model,           f'{model_dir}/decision_tree_model.pkl')
    joblib.dump(nb_model,           f'{model_dir}/naive_bayes_model.pkl')
    joblib.dump(scaler,             f'{model_dir}/scaler.pkl')
    joblib.dump(le,                 f'{model_dir}/label_encoder.pkl')
    joblib.dump(X.columns.tolist(), f'{model_dir}/feature_columns.pkl')
    joblib.dump(cv_scores,          f'{model_dir}/cv_scores.pkl')

    print('Model berhasil disimpan.')
    return dt_model, nb_model, scaler, le, X_train, X_test, y_train, y_test, cv_scores


if __name__ == '__main__':
    train_and_save()