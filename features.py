import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def engineer_features():
    print("Loading clean data...")
    df = pd.read_csv("data/clean.csv")
    
    print("Running VADER sentiment on all reviews... (1-2 min)")
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['review_text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )

    # Review-level features
    df['rating_normalized'] = (df['rating'] - 3) / 2
    df['mismatch_score'] = df['rating_normalized'] - df['sentiment_score']
    df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
    
    # Post-purchase pollution signals at review level
    pollution_phrases = [
        'already bought', 'already own', 'bought this before',
        'purchased before', 'own this', 'bought again',
        'repurchase', 're-purchase', 'bought it again',
        'second time buying', 'third time', 'keep buying',
        'bought multiple', 'already have'
    ]
    df['pollution_signal'] = df['review_text'].apply(
        lambda x: int(any(phrase in str(x).lower() for phrase in pollution_phrases))
    )

    print(f"\nReviews with pollution signal: {df['pollution_signal'].sum()}")
    print(f"Pollution rate: {df['pollution_signal'].mean():.1%}")

    # ── PRODUCT-LEVEL AGGREGATION ────────────────────────────────────
    print("\nAggregating to product level...")
    product_df = df.groupby('asin').agg(
        product_name    = ('product_name', 'first'),
        brand           = ('brand', 'first'),
        category        = ('category', 'first'),
        avg_rating      = ('rating', 'mean'),
        avg_sentiment   = ('sentiment_score', 'mean'),
        avg_mismatch    = ('mismatch_score', 'mean'),
        rating_variance = ('rating', 'var'),
        sentiment_variance = ('sentiment_score', 'var'),
        avg_review_length  = ('review_length', 'mean'),
        total_reviews   = ('rating', 'count'),
        pollution_count = ('pollution_signal', 'sum'),
        pct_1star       = ('rating', lambda x: (x == 1).mean()),
        pct_5star       = ('rating', lambda x: (x == 5).mean()),
        recommend_rate  = ('do_recommend', 'mean'),
    ).reset_index()

    product_df['pollution_rate'] = product_df['pollution_count'] / product_df['total_reviews']

    # ── TARGET LABEL ─────────────────────────────────────────────────
    # Pollution product = 4+ star avg rating BUT avg sentiment below 0.3
    # These are products Amazon keeps recommending despite lukewarm reality
    product_df['is_pollution_product'] = (
        (product_df['avg_rating'] >= 4.0) &
        (product_df['avg_sentiment'] < 0.3)
    ).astype(int)

    print(f"\nTotal products: {len(product_df)}")
    print(f"Pollution products (4+ stars but low sentiment): {product_df['is_pollution_product'].sum()}")
    print(f"\nProduct-level features:\n{product_df[['product_name', 'avg_rating', 'avg_sentiment', 'avg_mismatch', 'is_pollution_product']].to_string()}")

    df.to_csv("data/reviews_features.csv", index=False)
    product_df.to_csv("data/products_features.csv", index=False)
    print("\nSaved both feature files.")
    
    return df, product_df

if __name__ == "__main__":
    df, product_df = engineer_features()