import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Download the required lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

# 2. Load the cleaned dataset
# UPDATE: Ensure this matches the file name in your sidebar exactly
file_path = 'twitter_racism_parsed_dataset (1).csv'

try:
    df = pd.read_csv(file_path)
    print(f"'{file_path}' loaded successfully!")python -m pip install pandas matplotlib seaborn nltk
except FileNotFoundError:
    print(f"Error: '{file_path}' file not found. Check your file name in the sidebar.")
    exit()

# 3. Define the column to analyze
column_to_analyze = 'text'

if column_to_analyze not in df.columns:
    print(f"Error: Target column '{column_to_analyze}' not found.")
    print("Available columns are:", df.columns.tolist())
    exit()

# 4. Initialize the VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_category(text_data):
    if pd.isna(text_data):
        return 'Neutral'
    score = sia.polarity_scores(str(text_data))
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

print("Running sentiment analysis...")
df['Sentiment'] = df[column_to_analyze].apply(get_sentiment_category)