import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/charts", exist_ok=True)

def analyze_products():
    product_df = pd.read_csv("data/products_features.csv")

    # Pollution score: higher = more polluted
    # Combines mismatch, low sentiment, high rating
    product_df['pollution_score'] = (
        product_df['avg_mismatch'] * 0.5 +
        (1 - product_df['avg_sentiment']) * 0.3 +
        (product_df['avg_rating'] / 5) * 0.2
    ).round(3)

    product_df = product_df.sort_values('pollution_score', ascending=False)

    # Clean product names for display
    product_df['short_name'] = product_df['product_name'].apply(
        lambda x: str(x)[:40].strip() + '...' if len(str(x)) > 40 else str(x)
    )

    print("=== TOP 10 POLLUTION PRODUCTS ===")
    print(product_df[['short_name', 'avg_rating', 'avg_sentiment',
                       'avg_mismatch', 'pollution_score', 'is_pollution_product']].head(10).to_string())

    print(f"\n=== KEY FINDINGS ===")
    print(f"Total products analyzed: {len(product_df)}")
    print(f"Pollution products (4+ stars, low sentiment): {product_df['is_pollution_product'].sum()}")
    print(f"Avg rating across all products: {product_df['avg_rating'].mean():.2f}")
    print(f"Avg sentiment across all products: {product_df['avg_sentiment'].mean():.2f}")
    print(f"Products where sentiment < 0.3 but rating > 4: {((product_df['avg_sentiment'] < 0.3) & (product_df['avg_rating'] > 4)).sum()}")

    # Chart 1: Rating vs Sentiment scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#D85A30' if x == 1 else '#534AB7' for x in product_df['is_pollution_product']]
    scatter = ax.scatter(
        product_df['avg_rating'],
        product_df['avg_sentiment'],
        c=colors,
        s=product_df['total_reviews'] * 2,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    ax.axhline(0.3, color='red', linestyle='--', linewidth=1, label='Sentiment threshold (0.3)')
    ax.axvline(4.0, color='orange', linestyle='--', linewidth=1, label='4-star filter line')
    ax.set_xlabel('Average Star Rating', fontsize=12)
    ax.set_ylabel('Average Sentiment Score', fontsize=12)
    ax.set_title("Amazon's 4-Star Filter vs Real Sentiment\n(Red = Pollution Product, Size = Review Count)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.fill_betweenx([0, 0.3], [4, 4], [5, 5], alpha=0.08, color='red', label='Danger zone')
    plt.tight_layout()
    plt.savefig('notebooks/charts/rating_vs_sentiment.png', dpi=150)
    plt.close()
    print("\nSaved chart 1: rating_vs_sentiment.png")

    # Chart 2: Top 10 products by pollution score
    top10 = product_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ['#D85A30' if x == 1 else '#534AB7' for x in top10['is_pollution_product']]
    bars = ax.barh(top10['short_name'], top10['pollution_score'], color=bar_colors, edgecolor='white')
    ax.set_xlabel('Pollution Score', fontsize=12)
    ax.set_title('Top 10 Products by Pollution Score\n(Red = confirmed pollution product)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('notebooks/charts/top_pollution_products.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart 2: top_pollution_products.png")

    product_df.to_csv("data/products_scored.csv", index=False)
    print("\nSaved scored products to data/products_scored.csv")

    return product_df

if __name__ == "__main__":
    df = analyze_products()