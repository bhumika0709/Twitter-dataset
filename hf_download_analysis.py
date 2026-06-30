import pandas as pd
import matplotlib
# Prevent backend UI crashes on Windows
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Download necessary NLTK lexicon for VADER
nltk.download('vader_lexicon')

# List of the 5 datasets to scan
files_to_check = [
    'cleaned_dataset.csv', 
    'sonucduygu.csv', 
    'twitter_racism_parsed_dataset (1).csv',
    'Tesla.csv',
    'train.csv'
]

combined_dfs = []

print("--- Step 1: Scanning and combining all 5 datasets ---")
for file_name in files_to_check:
    path1 = file_name
    path2 = os.path.join('Combined datasets', file_name)
    
    actual_path = None
    if os.path.exists(path1):
        actual_path = path1
    elif os.path.exists(path2):
        actual_path = path2
        
    if actual_path:
        try:
            temp_df = pd.read_csv(actual_path, on_bad_lines='skip', engine='python')
            
            # Dynamically look for the text/tweet column
            text_col = None
            for col in ['tweet', 'text', 'annotation', 'Tweet', 'Text']:
                if col in temp_df.columns:
                    text_col = col
                    break
            
            # Fallback to the first column if no standard name matches
            if not text_col:
                text_col = temp_df.columns[0]
            
            # Clean missing rows and standardize column naming
            df_extracted = temp_df[[text_col]].dropna()
            df_extracted.columns = ['Cleaned_Text']
            combined_dfs.append(df_extracted)
            
            parent_dir = os.path.dirname(actual_path) or 'root'
            print(f"✅ Successfully loaded from '{parent_dir}': {file_name} ({len(df_extracted)} rows)")
            
        except Exception as e:
            print(f"⚠️ Could not read {file_name}: {e}")
    else:
        print(f"❌ File not found in root or 'Combined datasets' folder: {file_name}")

# Merge all valid dataframes into one
if combined_dfs:
    final_df = pd.concat(combined_dfs, ignore_index=True)
    print(f"\n🔥 Total Combined Rows across all datasets: {len(final_df)}")
else:
    print("❌ Error: No CSV files could be loaded! Please check your folder structure.")
    exit()

print("\n--- Step 2: Running Sentiment Analysis on Combined Data ---")
sia = SentimentIntensityAnalyzer()

def get_sentiment(text_data):
    score = sia.polarity_scores(str(text_data))
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

final_df['Sentiment'] = final_df['Cleaned_Text'].apply(get_sentiment)

print("\n--- Step 3: Generating Combined Graph ---")
plt.figure(figsize=(9, 6))
sns.set_theme(style="whitegrid")
sentiment_counts = final_df['Sentiment'].value_counts()

# Create structured bar chart
sns.barplot(
    x=sentiment_counts.index, 
    y=sentiment_counts.values, 
    palette={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
)

plt.title('Overall Combined Sentiment Analysis of All 5 Datasets', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sentiment Category', fontsize=12, labelpad=10)
plt.ylabel('Total Count', fontsize=12, labelpad=10)

# Display data counts dynamically above bars
for i, value in enumerate(sentiment_counts.values):
    plt.text(i, value + (max(sentiment_counts.values) * 0.01), f"{value:,}", ha='center', fontweight='bold')

plt.tight_layout()

# Save output image
output_img = 'all_datasets_combined_graph.png'
plt.savefig(output_img, dpi=300)
print(f"\n🎉 SUCCESS! Grand combined graph saved as: '{output_img}'")