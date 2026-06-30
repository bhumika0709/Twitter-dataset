 FeaturesMulti-Dataset Support: Cleans and processes multiple sources (cleaned_dataset.csv, sonucduygu.csv, twitter_racism_parsed_dataset (1).csv, Tesla.csv, and train.csv).
 VADER Sentiment Engine: Categorizes texts into Positive, Neutral, and Negative using NLTK.
 Automated Visualizations: Saves high-resolution distribution graphs automatically as .png files.Metadata Profiling: Tracks tweet length distributions and dataset column types.
 Setup & ExecutionInstallationBashpython -m pip install pandas matplotlib seaborn nltk
Run ProjectBashpython- hf_download_analysis.py
Sentiment ThresholdsPositive: Compound score $\ge 0.05$Negative: Compound score $\le -0.05$Neutral: Score between $-0.05$ and $0.05$
