import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the required VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
try:
    df = pd.read_csv('Tesla.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Make sure 'Tesla.csv' is in the same folder as this script.")
    exit()

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def get_sentiment_category(tweet):
    if pd.isna(tweet):
        return 'Neutral'
    score = sia.polarity_scores(str(tweet))
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

print("Analyzing sentiments...")
df['Sentiment'] = df['tweet'].apply(get_sentiment_category)

# Plot the Graph
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")
sentiment_counts = df['Sentiment'].value_counts()

sns.barplot(
    x=sentiment_counts.index, 
    y=sentiment_counts.values, 
    palette={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
)

plt.title('Sentiment Analysis of Tesla Tweets', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Sentiment Category', fontsize=12, labelpad=10)
plt.ylabel('Number of Tweets', fontsize=12, labelpad=10)

for i, value in enumerate(sentiment_counts.values):
    plt.text(i, value + (max(sentiment_counts.values) * 0.01), str(value), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_analysis_plot.png', dpi=300)
print("Graph saved as 'sentiment_analysis_plot.png'!")
plt.show()