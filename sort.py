# This script reads two CSV files, merges them, and creates a new CSV with one-hot columns for each label.
# It marks 'x' if the row's label matches the column, else leaves blank.
# Update the filenames as needed.

import pandas as pd

# List your CSV filenames here
csv_files = [
	'vietnamese_bias_chatgpt.csv',
	# Add your second CSV filename here
]

# List of all possible labels
labels = [
	'appearance_derogation',
	'gender_stereotype',
	'neutral',
	'regional_bias',
	'socioeconomic_occupation_bias',
]

# Read and concatenate all CSVs
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# For each label, create a new column
for label in labels:
	df[label] = df['label'].apply(lambda x: 'x' if x == label else '')

# Drop the original label column
df = df.drop(columns=['label'])

# Save to a new CSV
df.to_csv('vietnamese_bias_chatgpt_merged_onehot_labels.csv', index=False)
print('Saved vietnamese_bias_chatgpt_merged_onehot_labels.csv')
