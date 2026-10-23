

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="Findly | Smart Shopping",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f8fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #172554;
    letter-spacing: -1px;
}

.brand span {
    color: #6366f1;
}

.top-note {
    color: #64748b;
    font-size: 0.85rem;
    text-align: right;
    padding-top: 0.6rem;
}

.hero {
    background: linear-gradient(125deg, #172554 0%, #3730a3 65%, #6366f1 100%);
    padding: 42px 38px;
    border-radius: 24px;
    color: white;
    margin-top: 20px;
    margin-bottom: 26px;
}

.eyebrow {
    color: #c7d2fe;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.75rem;
    font-weight: 700;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    color: white;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.15;
    letter-spacing: -1.5px;
    margin: 12px 0;
}

.hero p {
    color: #e0e7ff;
    font-size: 1rem;
    max-width: 580px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    color: #172554;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.muted {
    color: #64748b;
    font-size: 0.9rem;
}

.product-card {
    background: white;
    border: 1px solid #e8eaf2;
    border-radius: 18px;
    padding: 15px;
    margin-bottom: 18px;
    min-height: 100%;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.035);
}

.product-image {
    height: 190px;
    width: 100%;
    object-fit: contain;
    background: #f8fafc;
    border-radius: 12px;
    padding: 12px;
}

.product-name {
    color: #1e293b;
    font-size: 0.96rem;
    font-weight: 700;
    line-height: 1.5;
    margin-top: 14px;
    min-height: 68px;
}

.product-category {
    color: #64748b;
    font-size: 0.77rem;
    margin-top: 5px;
    min-height: 35px;
}

.product-price {
    color: #4338ca;
    font-size: 1.25rem;
    font-weight: 800;
    margin-top: 12px;
}

.product-rating {
    color: #b45309;
    font-size: 0.85rem;
    margin-top: 6px;
}

.score {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    padding: 5px 9px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-top: 10px;
}

div[data-testid="stTextInput"] input {
    border: 1px solid #dbe1f0;
    border-radius: 12px;
    min-height: 48px;
    background: white;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px #e0e7ff;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #dbe1f0;
    color: #3730a3;
    background: white;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: #6366f1;
    color: #4338ca;
}

div[data-testid="stLinkButton"] a {
    border-radius: 10px;
    font-weight: 700;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    padding-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    data = pd.read_csv("data/raw/amazon.csv")

    data = data.dropna(subset=["product_name"]).reset_index(drop=True)

    for column in ["about_product", "category", "img_link",
                   "discounted_price", "rating", "product_link"]:
        if column not in data.columns:
            data[column] = ""

        data[column] = data[column].fillna("").astype(str)

    data["combined_features"] = (
        data["category"] + " " + data["about_product"]
    )

    return data


# ---------- BUILD RECOMMENDER ----------
@st.cache_resource
def create_similarity(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(matrix)


df = load_data()
similarity_matrix = create_similarity(df["combined_features"])


# ---------- HELPERS ----------
def format_price(value):
    value = str(value).strip()

    if not value:
        return "Price unavailable"

    return value


def show_product(product, score=None):
    image_url = str(product["img_link"]).strip()
    name = str(product["product_name"])
    category = str(product["category"])
    price = format_price(product["discounted_price"])
    rating = str(product["rating"]).strip()
    link = str(product["product_link"]).strip()

    with st.container(border=True):
        if image_url.startswith("http"):
            st.markdown(
                f'<img class="product-image" src="{image_url}" '
                f'alt="Product image" />',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="product-image" '
                'style="display:flex;align-items:center;'
                'justify-content:center;font-size:3rem;">🛍️</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="product-name">{name}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="product-category">{category}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="product-price">{price}</div>',
            unsafe_allow_html=True
        )

        if rating:
            st.markdown(
                f'<div class="product-rating">★ {rating} / 5</div>',
                unsafe_allow_html=True
            )

        if score is not None:
            st.markdown(
                f'<span class="score">{score:.0%} similarity</span>',
                unsafe_allow_html=True
            )

        if link.startswith("http"):
            st.link_button(
                "View product ↗",
                link,
                use_container_width=True
            )


# ---------- HEADER ----------
header_left, header_right = st.columns([3, 2])

with header_left:
    st.markdown(
        '<div class="brand">findly<span>.</span></div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="top-note">SMART DISCOVERY · PERSONALIZED PICKS</div>',
        unsafe_allow_html=True
    )


# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">Your smarter way to shop</div>
    <h1>Find your next<br>favourite product.</h1>
    <p>Discover products similar to what you love, powered by
    machine learning and product descriptions.</p>
</div>
""", unsafe_allow_html=True)


# ---------- SEARCH ----------
st.markdown(
    '<div class="section-title">What are you looking for?</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="muted">Search for a product to discover similar picks.</div>',
    unsafe_allow_html=True
)

search = st.text_input(
    "Product search",
    placeholder="Try 'USB cable', 'wireless mouse' or 'headphones'...",
    label_visibility="collapsed"
)


# ---------- RECOMMENDATIONS ----------
if search.strip():
    matches = df[
        df["product_name"].str.contains(
            search.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        st.warning(
            "We couldn't find that product. Try a shorter product name "
            "or a different keyword."
        )
    else:
        selected_index = int(matches.index[0])
        selected = df.loc[selected_index]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Your selected product</div>',
            unsafe_allow_html=True
        )

        show_product(selected)

        scores = list(enumerate(similarity_matrix[selected_index]))
        scores.sort(key=lambda item: item[1], reverse=True)

        # Do not recommend the selected product itself
        scores = [
            (index, score)
            for index, score in scores
            if index != selected_index
        ]

        st.markdown("---")

        title_col, count_col = st.columns([3, 1])

        with title_col:
            st.markdown(
                '<div class="section-title">Recommended for you</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="muted">Similar products ranked by '
                'description and category.</div>',
                unsafe_allow_html=True
            )

        with count_col:
            st.markdown(
                '<div style="text-align:right;color:#6366f1;'
                'font-weight:700;padding-top:12px;">TOP PICKS</div>',
                unsafe_allow_html=True
            )

        columns = st.columns(3)

        for position, (index, score) in enumerate(scores[:6]):
            with columns[position % 3]:
                show_product(df.loc[index], score)

else:
    st.markdown("---")

    st.markdown(
        '<div class="section-title">Ready to discover?</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Enter a product name above to get '
        'recommendations tailored to its description.</div>',
        unsafe_allow_html=True
    )

    st.markdown("")
    st.info(
        "💡 Tip: Start with a product name that appears in your dataset."
    )


# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    FINDLY · A product recommendation system built with Python, Streamlit, and scikit-learn.<br>
    <a href="https://github.com/fndly/product-recommendation-system" target="_blank">View the source code</a>
</div>
""", unsafe_allow_html=True)
