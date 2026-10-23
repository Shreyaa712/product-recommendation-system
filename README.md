##Product Recommendation System

A content-based recommendation system built using Python to recommend similar products based on their descriptions.

### Key Features
- **Content-based filtering** using product descriptions and categories.
- **TF-IDF vectorization** to convert text into numerical features.
- **Cosine similarity** to identify and recommend similar products.
- Displays product images, prices, ratings, and product links.

### User Flow
1. Search for a product by name.
2. View the selected product.
3. Explore similar product recommendations.
4. Compare products and visit product links.

### Tech Stack
Python, Pandas, Scikit-learn, Streamlit

### Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

**Dataset:** Amazon product dataset from Kaggle. Place `amazon.csv` in `data/raw/` before running the app.
