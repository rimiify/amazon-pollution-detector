import pandas as pd

def load_data():
    print("Loading dataset...")
    df = pd.read_csv("data/1429_1.csv", low_memory=False)

    keep_cols = [
        'name',
        'brand',
        'categories',
        'reviews.rating',
        'reviews.text',
        'reviews.title',
        'reviews.numHelpful',
        'reviews.doRecommend',
        'asins',
    ]

    df = df[keep_cols].copy()
    df.columns = [
        'product_name',
        'brand',
        'category',
        'rating',
        'review_text',
        'review_title',
        'helpful_votes',
        'do_recommend',
        'asin',
    ]

    df = df.dropna(subset=['review_text', 'rating'])
    df['rating'] = df['rating'].astype(float)
    df['helpful_votes'] = df['helpful_votes'].fillna(0).astype(int)
    df['do_recommend'] = df['do_recommend'].fillna(True)
    df['review_text'] = df['review_text'].astype(str)

    print(f"Clean shape: {df.shape}")
    print(f"\nRating distribution:\n{df['rating'].value_counts().sort_index()}")
    print(f"\nUnique products: {df['asin'].nunique()}")

    df.to_csv("data/clean.csv", index=False)
    print("\nSaved to data/clean.csv")
    return df

if __name__ == "__main__":
    df = load_data()