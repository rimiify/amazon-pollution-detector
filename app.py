import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Amazon Pollution Detector",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/products_scored.csv")

@st.cache_data
def load_reviews():
    return pd.read_csv("data/reviews_features.csv")

product_df = load_data()
reviews_df = load_reviews()
analyzer = SentimentIntensityAnalyzer()

# Clean product names
product_df['short_name'] = product_df['product_name'].apply(
    lambda x: str(x)[:60].strip().replace('\r\n', ' ').replace(',,,', '').strip()
)
product_df['short_name'] = product_df['short_name'].replace('nan', 'Unknown Product')

# ── HEADER ──────────────────────────────────────────────────────────
st.title("🔍 Amazon Recommendation Pollution Detector")
st.markdown("""
**The problem:** Amazon keeps recommending products customers already own or have moved past.  
This tool audits which products show the highest gap between star rating and actual sentiment —  
products Amazon's algorithm keeps pushing despite lukewarm customer reality.
""")
st.divider()

# ── METRICS ROW ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Products Analyzed", len(product_df))
with col2:
    st.metric("Avg Star Rating", f"{product_df['avg_rating'].mean():.2f} ⭐")
with col3:
    st.metric("Avg Sentiment Score", f"{product_df['avg_sentiment'].mean():.2f}")
with col4:
    pollution_count = product_df['is_pollution_product'].sum()
    st.metric("Pollution Products Found", f"{pollution_count} 🚨")

st.divider()

# ── TABS ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Product Audit", "🔬 Analyze a Review", "📈 Charts"])

# ── TAB 1: PRODUCT AUDIT TABLE ───────────────────────────────────────
with tab1:
    st.subheader("All Products Ranked by Pollution Score")
    st.caption("Higher pollution score = high star rating but low actual sentiment. These are products Amazon keeps recommending despite customer dissatisfaction.")

    display_df = product_df[[
        'short_name', 'avg_rating', 'avg_sentiment',
        'avg_mismatch', 'pollution_score', 'total_reviews', 'is_pollution_product'
    ]].copy()

    display_df.columns = [
        'Product', 'Avg Rating', 'Avg Sentiment',
        'Mismatch Score', 'Pollution Score', 'Total Reviews', 'Is Pollution Product'
    ]
    display_df = display_df.sort_values('Pollution Score', ascending=False).reset_index(drop=True)
    display_df['Avg Rating'] = display_df['Avg Rating'].round(2)
    display_df['Avg Sentiment'] = display_df['Avg Sentiment'].round(3)
    display_df['Mismatch Score'] = display_df['Mismatch Score'].round(3)
    display_df['Pollution Score'] = display_df['Pollution Score'].round(3)
    display_df['Is Pollution Product'] = display_df['Is Pollution Product'].apply(
        lambda x: '🚨 Yes' if x == 1 else '✅ No'
    )

    st.dataframe(display_df, use_container_width=True, height=400)

    st.divider()
    st.subheader("🚨 Confirmed Pollution Products")
    pollution_products = product_df[product_df['is_pollution_product'] == 1][
        ['short_name', 'avg_rating', 'avg_sentiment', 'pollution_score', 'total_reviews']
    ].sort_values('pollution_score', ascending=False)

    for _, row in pollution_products.iterrows():
        name = str(row['short_name']).replace('\r\n', ' ').replace(',,,', '').strip()
        with st.expander(f"🚨 {name[:70]}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Star Rating", f"{row['avg_rating']:.2f} ⭐")
            c2.metric("Actual Sentiment", f"{row['avg_sentiment']:.3f}")
            c3.metric("Pollution Score", f"{row['pollution_score']:.3f}")
            st.caption(f"Based on {int(row['total_reviews'])} reviews")

# ── TAB 2: ANALYZE A REVIEW ──────────────────────────────────────────
with tab2:
    st.subheader("Analyze Any Amazon Review")
    st.caption("Paste a review to see its sentiment score and whether it contributes to recommendation pollution.")

    review_input = st.text_area(
        "Paste review text here",
        placeholder="e.g. I already bought this last year. It works fine I guess...",
        height=150
    )
    rating_input = st.slider("Star rating given", 1.0, 5.0, 4.0, step=0.5)

    if st.button("Analyze Review", type="primary"):
        if not review_input.strip():
            st.warning("Please paste a review first.")
        else:
            sentiment = analyzer.polarity_scores(review_input)['compound']
            rating_norm = (rating_input - 3) / 2
            mismatch = rating_norm - sentiment

            pollution_phrases = [
                'already bought', 'already own', 'bought this before',
                'purchased before', 'own this', 'bought again',
                'repurchase', 're-purchase', 'bought it again',
                'second time buying', 'third time', 'keep buying',
                'bought multiple', 'already have'
            ]
            has_pollution = any(p in review_input.lower() for p in pollution_phrases)

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Sentiment Score", f"{sentiment:.3f}")
            m2.metric("Mismatch Score", f"{mismatch:.3f}")
            m3.metric("Pollution Signal", "🚨 Detected" if has_pollution else "✅ None")

            if mismatch > 0.3 and rating_input >= 4:
                st.error("🚨 This review contributes to recommendation pollution — high stars but lukewarm sentiment")
            elif has_pollution:
                st.warning("⚠️ Post-purchase language detected — customer may already own this product")
            else:
                st.success("✅ Review sentiment aligns with star rating")

# ── TAB 3: CHARTS ────────────────────────────────────────────────────
with tab3:
    st.subheader("Rating vs Sentiment — Where Amazon's Filter Breaks")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#D85A30' if x == 1 else '#534AB7'
              for x in product_df['is_pollution_product']]
    ax.scatter(
        product_df['avg_rating'],
        product_df['avg_sentiment'],
        c=colors,
        s=product_df['total_reviews'] * 2,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    ax.axhline(0.3, color='red', linestyle='--', linewidth=1.5,
               label='Sentiment threshold (0.3)')
    ax.axvline(4.0, color='orange', linestyle='--', linewidth=1.5,
               label="Amazon's 4-star filter")
    ax.fill_betweenx([0, 0.3], [4, 4], [5, 5],
                     alpha=0.1, color='red')
    ax.set_xlabel('Average Star Rating', fontsize=12)
    ax.set_ylabel('Average Sentiment Score', fontsize=12)
    ax.set_title("Amazon's 4-Star Filter vs Real Sentiment\n"
                 "(Red = Pollution Product, Size = Review Count)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        "Products in the bottom-right zone (high rating, low sentiment) "
        "are passing Amazon's star filter despite customer dissatisfaction. "
        "These are recommendation pollution candidates."
    )

st.divider()
st.caption(
    "Built by Rimi | Aliah University CSE '27 | "
    "github.com/rimiify | "
    "Trained on 34,660 Amazon Electronics reviews"
)