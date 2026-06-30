import pandas as pd
import matplotlib.pyplot as plt

# --- DATASET 2 ANALYSIS ---
file_name = 'sonucduygu.csv'

try:
    # 1. Load Dataset 2 with the correct semicolon separator
    df = pd.read_csv(file_name, sep=';')
    df.columns = df.columns.str.strip()
    
    # Extract the exact first column name dynamically
    raw_text_column = df.columns[0]

    # Task ① and ②: Printing Samples and Features
    print("\n==========================================")
    print(f"📊 ANALYSIS FOR DATASET 2: {file_name}")
    print("==========================================")
    print(f"① No. of Samples (Total Rows): {len(df)}")
    print("\n② Features and Data Types:")
    print(df.dtypes)
    print("==========================================\n")
    
    # 2. Visualization (Tweet Length Distribution)
    df['tweet_length'] = df[raw_text_column].astype(str).apply(len)
    
    plt.figure(figsize=(6, 4))
    plt.hist(df['tweet_length'], bins=20, color='#4ade80', edgecolor='black')
    plt.title(f'Dataset 2: Tweet Length Distribution')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    print("Opening Graph Window for Dataset 2 now...")
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: '{file_name}' file not found in your folder!") 