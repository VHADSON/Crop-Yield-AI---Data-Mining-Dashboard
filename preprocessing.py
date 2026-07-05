import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(df):
    # Drop kolom tidak penting
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=unnamed_cols, errors='ignore')

    # Hapus duplikat
    df = df.drop_duplicates()

    # Rename kolom agar konsisten
    df.columns = [
        'Area',
        'Item',
        'Year',
        'hg/ha_yield',
        'average_rain_fall_mm_per_year',
        'pesticides_tonnes',
        'avg_temp'
    ]

    # Cleaning text
    df['Area'] = df['Area'].apply(clean_text)
    df['Item'] = df['Item'].apply(clean_text)

    # Diskritisasi target
    df['Yield_Class'] = pd.qcut(
        df['hg/ha_yield'],
        q=3,
        labels=['Rendah', 'Sedang', 'Tinggi']
    )

    # Label Encoding untuk target
    le = LabelEncoder()
    le.fit(['Rendah', 'Sedang', 'Tinggi'])

    # One hot encoding
    df_encoded = pd.get_dummies(df, columns=['Area', 'Item'])

    # Scaling numerik
    scaler = StandardScaler()

    numeric_cols = [
        'Year',
        'average_rain_fall_mm_per_year',
        'pesticides_tonnes',
        'avg_temp'
    ]

    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    return df, df_encoded, scaler, le
