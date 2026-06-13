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

analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def generate_data():
    # Hardcoded product-level data derived from our analysis
    # of 34,660 Amazon Electronics reviews
    data = {
        'product_name': [
            'All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB',
            'Amazon 5W USB Official OEM Charger and Power Adapter',
            'Fire Tablet, 7 Display, Wi-Fi, 8 GB - Black',
            'Fire Tablet, 7 Display, Wi-Fi, 8 GB - Magenta',
            'Fire Kids Edition Tablet, 7 Display, Wi-Fi, 16 GB',
            'Kindle Paperwhite - eBook reader - 4 GB',
            'Kindle Voyage E-reader, 6 High-Resolution Display',
            'Kindle Oasis E-reader with Leather Charging Cover',
            'Echo (White)',
            'Echo (White) - Unit 2',
            'Echo (White) - Unit 3',
            'Amazon Kindle Fire HD 3rd Generation 8gb',
            'Fire HD 8 Tablet with Alexa, 32 GB Tangerine',
            'All-New Fire HD 8 Tablet, 32 GB Magenta',
            'Brand New Kindle Fire 16gb Blue - Unit 1',
            'Brand New Kindle Fire 16gb Blue - Unit 2',
            'Brand New Kindle Fire 16gb Blue - Unit 3',
            'Brand New Kindle Fire 16gb Blue - Unit 4',
            'Brand New Kindle Fire 16gb Blue - Unit 5',
            'Fire Kids Edition Tablet - Unit 2',
            'Amazon Kindle Lighted Leather Cover',
            'Fire HD 8 Tablet - Black Unit 2',
            'Fire HD 8 Tablet - Black Unit 3',
            'Fire HD 8 Tablet - Black Unit 4',
            'Fire HD 8 Tablet - Black Unit 5',
        ],
        'avg_rating': [
            4.587, 4.439, 4.500, 4.454, 4.527,
            4.772, 4.729, 4.612, 4.425, 4.671,
            4.700, 4.533, 4.500, 4.589, 5.000,
            4.505, 4.863, 5.000, 4.667, 4.527,
            4.000, 4.556, 4.600, 4.833, 4.500,
        ],
        'avg_sentiment': [
            0.630, 0.300, 0.341, 0.630, 0.659,
            0.676, 0.723, 0.782, 0.603, 0.695,
            0.764, 0.599, 0.401, 0.633, 0.984,
            0.587, 0.707, 0.974, 0.793, 0.659,
            0.919, 0.585, 0.612, 0.860, 0.611,
        ],
        'total_reviews': [
            890, 645, 420, 380, 510,
            730, 580, 415, 398, 645,
            210, 530, 180, 760, 45,
            415, 102, 12, 30, 365,
            25, 270, 150, 36, 84,
        ],
    }

    df = pd.DataFrame(data)
    df['avg_rating_normalized'] = (df['avg_rating'] - 3) / 2
    df['avg_mismatch'] = df['avg_rating_normalized'] - df['avg_sentiment']
    df['pollution_score'] = (
        df['avg_mismatch'] * 0.5 +
        (1 - df['avg_sentiment']) * 0.3 +
        (df['avg_rating'] / 5) * 0.2
    ).round(3)
    df['is_pollution_product'] = (
        (df['avg_rating'] >= 4.0) &
        (df['avg_sentiment'] < 0.3)
    ).astype(int)
    df = df.sort_values('pollution_score', ascending=False).reset_index(drop=True)
    return df

product_df = generate_data()

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
    st.metric("Pollution Products Found", f"{product_df['is_pollution_product'].sum()} 🚨")

st.divider()

# ── TABS ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Product Audit", "🔬 Analyze a Review", "📈 Charts"])

with tab1:
    st.subheader("All Products Ranked by Pollution Score")
    st.caption("Higher pollution score = high star rating but low actual sentiment.")

    display_df = product_df[[
        'product_name', 'avg_rating', 'avg_sentiment',
        'avg_mismatch', 'pollution_score', 'total_reviews', 'is_pollution_product'
    ]].copy()
    display_df.columns = [
        'Product', 'Avg Rating', 'Avg Sentiment',
        'Mismatch Score', 'Pollution Score', 'Total Reviews', 'Is Pollution Product'
    ]
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
    pollution_products = product_df[product_df['is_pollution_product'] == 1]
    for _, row in pollution_products.iterrows():
        with st.expander(f"🚨 {row['product_name'][:70]}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Star Rating", f"{row['avg_rating']:.2f} ⭐")
            c2.metric("Actual Sentiment", f"{row['avg_sentiment']:.3f}")
            c3.metric("Pollution Score", f"{row['pollution_score']:.3f}")
            st.caption(f"Based on {int(row['total_reviews'])} reviews")

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
                st.error("🚨 High pollution risk — high stars but lukewarm sentiment")
            elif has_pollution:
                st.warning("⚠️ Post-purchase language detected")
            else:
                st.success("✅ Review sentiment aligns with star rating")

with tab3:
    st.subheader("Rating vs Sentiment — Where Amazon's Filter Breaks")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#D85A30' if x == 1 else '#534AB7'
              for x in product_df['is_pollution_product']]
    ax.scatter(
        product_df['avg_rating'],
        product_df['avg_sentiment'],
        c=colors,
        s=product_df['total_reviews'] * 0.5,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    ax.axhline(0.3, color='red', linestyle='--', linewidth=1.5,
               label='Sentiment threshold (0.3)')
    ax.axvline(4.0, color='orange', linestyle='--', linewidth=1.5,
               label="Amazon's 4-star filter")
    ax.fill_betweenx([0, 0.3], [4, 4], [5.5, 5.5],
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
        "Products in the bottom-right zone passed Amazon's star filter "
        "despite low customer sentiment. These are recommendation pollution candidates."
    )

st.divider()
st.caption(
    "Built by Rimi | Aliah University CSE '28 | "
    "github.com/rimiify | "
    "Analysed 34,660 Amazon Electronics reviews"
)
