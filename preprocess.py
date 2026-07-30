"""
Shared preprocessing logic for the Addiction Level model.
This must exactly mirror the steps performed in the training notebook
(Social_Media_Addiction___Mental_Wellbeing.ipynb) before the data reached
the scaler / model.
"""

import pandas as pd

# Exact column order the model was trained on (X.columns from the notebook)
FEATURE_COLUMNS = [
    'age', 'gender', 'affiliate_organization', 'platforms', 'avg_time_per_day',
    'use_without_purpose(1-5)', 'distracted(1-5)', 'restless(1-5)', 'distracted_easily(1-5)',
    'worries(1-5)', 'concentration(1-5)', 'compare_to_others(1-5)', 'compare_feelings(1-5)',
    'validation(1-5)', 'depressed(1-5)', 'daily_activity_fluctuate(1-5)', 'sleeping_issues(1-5)',
    'relationship_status_In a relationship', 'relationship_status_Married', 'relationship_status_Single',
    'occupation_status_Salaried Worker', 'occupation_status_School Student', 'occupation_status_University Student'
]

RELATIONSHIP_OPTIONS = ["Single", "Married", "In a relationship", "Divorced"]
OCCUPATION_OPTIONS = ["University Student", "Salaried Worker", "School Student", "Retired"]
AVG_TIME_OPTIONS = [
    "Less than an Hour", "Between 1 and 2 hours", "Between 2 and 3 hours",
    "Between 3 and 4 hours", "Between 4 and 5 hours", "More than 5 hours",
]
RATING_FIELDS = [
    'use_without_purpose(1-5)', 'distracted(1-5)', 'restless(1-5)', 'distracted_easily(1-5)',
    'worries(1-5)', 'concentration(1-5)', 'compare_to_others(1-5)', 'compare_feelings(1-5)',
    'validation(1-5)', 'depressed(1-5)', 'daily_activity_fluctuate(1-5)', 'sleeping_issues(1-5)',
]


def _gender_map(value):
    """Same rule as the notebook: anything not literally 'female' becomes 'Male'."""
    return 'Female' if str(value).strip().lower() == 'female' else 'Male'


def _count_parts(value):
    """Same rule as the notebook's count_affiliations / count_platforms."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    parts = str(value).split(',')
    count = 0
    for p in parts:
        p = p.strip().lower()
        if p != '' and p != 'n/a':
            count += 1
    return count


def preprocess_input(user_data: dict, le_gender, encoder_avg_time, scaler) -> pd.DataFrame:
    """
    Turn one raw user submission into a single-row, scaled DataFrame
    that model_addiction.predict() can consume directly.

    user_data expects these raw keys:
        age (int)
        gender (str)                      e.g. "Female"
        relationship_status (str)         one of RELATIONSHIP_OPTIONS
        occupation_status (str)           one of OCCUPATION_OPTIONS
        affiliate_organization (str)      comma separated raw text, or "N/A"
        platforms (str)                   comma separated raw text, or "N/A"
        avg_time_per_day (str)            one of AVG_TIME_OPTIONS
        <12 rating fields>                ints 1-5, keys listed in RATING_FIELDS
    """
    df_input = pd.DataFrame([user_data])

    # 1. Gender
    df_input['gender'] = df_input['gender'].apply(_gender_map)
    df_input['gender'] = le_gender.transform(df_input['gender'])

    # 2. Free-text counts
    df_input['affiliate_organization'] = df_input['affiliate_organization'].apply(_count_parts)
    df_input['platforms'] = df_input['platforms'].apply(_count_parts)

    # 3. Ordinal: avg_time_per_day
    df_input['avg_time_per_day'] = encoder_avg_time.transform(df_input[['avg_time_per_day']])

    # 4. One-hot: relationship_status / occupation_status (manual, matches drop_first=True)
    relationship = df_input.pop('relationship_status').iloc[0]
    occupation = df_input.pop('occupation_status').iloc[0]

    for col in ['relationship_status_In a relationship', 'relationship_status_Married', 'relationship_status_Single']:
        df_input[col] = 1 if col == f'relationship_status_{relationship}' else 0

    for col in ['occupation_status_Salaried Worker', 'occupation_status_School Student', 'occupation_status_University Student']:
        df_input[col] = 1 if col == f'occupation_status_{occupation}' else 0

    # 5. Column order must match training exactly
    df_input = df_input[FEATURE_COLUMNS]

    # 6. Scale (transform only, scaler is already fit)
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

    return df_scaled
