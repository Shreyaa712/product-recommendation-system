
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
df = pd.read_csv("data/raw/amazon.csv")

# 2. Handle missing values
df = df.dropna(subset=["product_name"]).reset_index(drop=True)
df["about_product"] = df["about_product"].fillna("")
df["category"] = df["category"].fillna("")

# 3. Combine category and description
df["combined_features"] = (
    df["category"].astype(str) + " " +
    df["about_product"].astype(str)
)

# 4. Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# 5. Calculate similarity scores
similarity_matrix = cosine_similarity(tfidf_matrix)

# 6. Recommend similar products
def recommend_products(product_name, number=5):

    matches = df[
        df["product_name"].str.contains(
            product_name, case=False, na=False, regex=False
        )
    ]

    if matches.empty:
        print("Product not found. Try another product name.")
        return

    product_index = matches.index[0]

    scores = list(enumerate(similarity_matrix[product_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Remove the selected product itself
    scores = [item for item in scores if item[0] != product_index]

    # Select the top recommendations
    top_products = scores[:number]

    print("\nSelected product:")
    print(df.loc[product_index, "product_name"])

    print("\nRecommended products:")

    for index, score in top_products:
        product = df.loc[index]

        print("\nProduct:", product["product_name"])
        print("Category:", product["category"])
        print("Price:", product["discounted_price"])
        print("Rating:", product["rating"])
        print("Similarity:", round(score, 3))
        print("Open product:", product["product_link"])


# 7. Test the recommendation system
recommend_products("USB cable")
