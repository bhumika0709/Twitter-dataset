import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Download the required lexicon for sentiment analysis
nltk.download('vader_lexicon')

# 2. Load the twitter racism dataset with error handling for malformed rows
# 2. Load the twitter racism dataset with error handling for malformed rows
try:
    df = pd.read_csv('twitter_racism_parsed_dataset (1).csv', on_bad_lines='skip', engine='python')
    print("Twitter Racism Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'twitter_racism_parsed_dataset (1).csv' file not found. Please check the file name.")
    exit()

# 3. Specify the target column
# As seen in your file preview, the column name is 'Text'
column_to_analyze = 'Text'

if column_to_analyze not in df.columns:
    print(f"Error: Target column '{column_to_analyze}' not found in dataset.")
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

print("Running sentiment analysis on racism dataset...")
df['Sentiment'] = df[column_to_analyze].apply(get_sentiment_category)

# 5. Generate and plot the graph
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")
sentiment_counts = df['Sentiment'].value_counts()

sns.barplot(
    x=sentiment_counts.index, 
    y=sentiment_counts.values, 
    palette={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
)

plt.title('Sentiment Analysis of Twitter Racism Dataset', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Sentiment Category', fontsize=12, labelpad=10)
plt.ylabel('Number of Tweets', fontsize=12, labelpad=10)

# Display counts on top of each bar
for i, value in enumerate(sentiment_counts.values):
    plt.text(i, value + (max(sentiment_counts.values) * 0.01), str(value), ha='center', fontweight='bold')

plt.tight_layout()

# Save the plot and display it
plt.savefig('racism_sentiment_plot.png', dpi=300)
print("Graph successfully saved as 'racism_sentiment_plot.png'!")
plt.show()