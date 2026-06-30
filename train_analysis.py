import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Download the required lexicon for sentiment analysis
nltk.download('vader_lexicon')

# 2. Load the train dataset with error handling for malformed rows
try:
    # 'on_bad_lines="skip"' will ignore rows that cause parsing errors
    df = pd.read_csv('train.csv', on_bad_lines='skip', engine='python')
    print("Train Dataset loaded successfully (some malformed rows might have been skipped)!")
except FileNotFoundError:
    print("Error: 'train.csv' file not found. Please check the file name.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the file: {e}")
    exit()

# 3. Identify the target column
# Checking common names like 'tweet' or 'text'. If different, change it here.
column_to_analyze = None
for col in ['tweet', 'text', 'Tweets', 'Text']:
    if col in df.columns:
        column_to_analyze = col
        break

if column_to_analyze is None:
    print("Error: Could not automatically find the text or tweet column.")
    print("Available columns in your train.csv are:", df.columns.tolist())
    exit()
else:
    print(f"Analyzing sentiment using the column: '{column_to_analyze}'")

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

print("Running sentiment analysis on train dataset...")
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

plt.title('Sentiment Analysis of Train Dataset', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Sentiment Category', fontsize=12, labelpad=10)
plt.ylabel('Number of Tweets / Texts', fontsize=12, labelpad=10)

# Display counts on top of each bar
for i, value in enumerate(sentiment_counts.values):
    plt.text(i, value + (max(sentiment_counts.values) * 0.01), str(value), ha='center', fontweight='bold')

plt.tight_layout()

# Save the plot and display it
plt.savefig('train_sentiment_plot.png', dpi=300)
print("Graph successfully saved as 'train_sentiment_plot.png'!")
plt.show()