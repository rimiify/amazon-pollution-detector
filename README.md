# Amazon Recommendation Pollution Detector

> **The insight:** Amazon's 4-star filter is broken.  
> Products pass the star filter but fail the sentiment test — and Amazon keeps recommending them anyway.

🚨 **[Live Demo](https://amazon-pollution-detector.streamlit.app/)** | Built with Python, VADER, Streamlit

---

## The Problem

Every Amazon customer filters by 4 stars and above assuming it means quality. But that filter is built on inflated ratings. A product can sit at 4.4 stars with hundreds of reviews and have near-neutral average sentiment — meaning customers aren't actually happy, they're just generous with stars.

Amazon's recommendation algorithm keeps pushing these products. That's recommendation pollution.

---

## The Hypothesis

If a product has 4+ star average rating but below-neutral review sentiment, it is a **pollution product** — passing Amazon's quality filter despite customer dissatisfaction.

**Finding:** Amazon's own 5W USB Charger is a confirmed pollution product — 4.4 stars, sentiment score of only 0.300. Customers rate it highly but write about it with near-neutral language.

---

## What This Tool Does

- Audits 25 Amazon Electronics products across 34,660 reviews
- Calculates average sentiment per product using VADER NLP
- Computes a **pollution score** combining mismatch, low sentiment, and high rating
- Flags confirmed pollution products — high stars, low actual sentiment
- Lets you paste any Amazon review and see its real sentiment vs star rating

---

## Approach

**1. Review-level sentiment extraction**  
VADER sentiment analysis on all 34,660 Electronics reviews — compound score from -1 to +1.

**2. Product-level aggregation**  
Grouped by product, calculated average rating, average sentiment, mismatch score, and rating variance.

**3. Pollution score formula**  
`pollution_score = (mismatch × 0.5) + (1 - sentiment × 0.3) + (rating/5 × 0.2)`

**4. Pollution label**  
Products with 4+ star average rating AND sentiment below 0.35 = confirmed pollution product.

---

## Key Finding

| Product | Avg Rating | Avg Sentiment | Pollution? |
|---|---|---|---|
| Amazon 5W USB Charger | 4.44 ⭐ | 0.300 | 🚨 Yes |
| Fire Tablet 7" Black | 4.50 ⭐ | 0.341 | 🚨 Yes |
| All-New Fire HD 8 Tablet | 4.59 ⭐ | 0.630 | ✅ No |

---

## Stack

Python • VADER Sentiment • Pandas • Matplotlib • Streamlit

---

## Dataset

[Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) — 34,660 Electronics reviews via Kaggle.

---

## Run Locally

```bash
git clone https://github.com/rimiify/amazon-pollution-detector.git
cd amazon-pollution-detector
pip install -r requirements.txt
streamlit run app.py
```

---

*Built by [Rimi](https://github.com/rimiify) — BTech CSE, Aliah University '28*  
*Amazon ML Summer School 2026 applicant*
