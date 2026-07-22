import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_JSON = 'data/dataset.json'
OUTPUT_DIR = 'data/analysis_experiment_004'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_JSON, 'r') as f:
    dataset = json.load(f)

records = []
for split, items in dataset.items():
    is_train = (split == 'train')
    group = 'Train (Sparse)' if is_train else 'Val+Test (Full)'
    for item in items:
        shape = item['shape']
        scan_vol = shape[0] * shape[1] * shape[2]
        for f_idx, cat in item.get('categories', {}).items():
            if f_idx not in item.get('pixels', {}):
                continue
            pixels = item['pixels'][f_idx]
            entity_count = item.get('entity_counts', {}).get(f_idx, 1)
            records.append({
                'Split': group,
                'Category': cat,
                'RelativeVolume': pixels / scan_vol,
                'EntityCount': entity_count,
                'TotalFindingsInScan': len(item['categories']),
                'Filename': item['name']
            })

df = pd.DataFrame(records)
print(f"Total finding records parsed: {len(df)}")

# 1. Relative Volume Boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Category', y='RelativeVolume', hue='Split')
plt.yscale('log')
plt.xticks(rotation=90)
plt.title('Distribution of Mask Volume Relative to Scan Volume')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'relative_volume_boxplot.png'), dpi=300)
plt.close()

# 2. Average Entity Count
plt.figure(figsize=(14, 8))
sns.barplot(data=df, x='Category', y='EntityCount', hue='Split')
plt.xticks(rotation=90)
plt.title('Average Entity Counts per Finding Category')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'average_entity_counts.png'), dpi=300)
plt.close()

# 3. Histogram of Entity Counts
plt.figure(figsize=(12, 6))
df_filtered = df[df['EntityCount'] <= 15]
sns.histplot(data=df_filtered, x='EntityCount', hue='Split', multiple='dodge', discrete=True, stat='proportion', common_norm=False, shrink=0.8)
plt.title('Distribution of Entity Counts per Finding (Normalized)')
plt.xlabel('Entity Count (Number of Instances)')
plt.ylabel('Proportion of Findings in Split')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'entity_counts_hist.png'), dpi=300)
plt.close()

# 4. Findings per Scan
scans_df = df.groupby(['Filename', 'Split'])['TotalFindingsInScan'].first().reset_index()
plt.figure(figsize=(12, 6))
scans_df_filtered = scans_df[scans_df['TotalFindingsInScan'] <= 20]
sns.histplot(data=scans_df_filtered, x='TotalFindingsInScan', hue='Split', multiple='dodge', discrete=True, stat='proportion', common_norm=False, shrink=0.8)
plt.title('Distribution of Findings Count per Scan (Normalized)')
plt.xlabel('Total Number of Findings in a Single Scan')
plt.ylabel('Proportion of Scans in Split')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'findings_per_scan.png'), dpi=300)
plt.close()

print("Descriptive statistics generated successfully.")
