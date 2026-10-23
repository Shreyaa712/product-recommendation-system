###Product Recommendation System
#  Findly — Product Recommendation System

**Find products that are similar to what you already love.**

Findly is a content-based product recommendation system built with Python. It analyzes product categories and descriptions to recommend similar items through a simple, interactive shopping interface.

## ✨ Features

- **Product search** — Search for a product by name.
- **Similar product recommendations** — Discover products using text similarity.
- **Product cards** — View product images, prices, and ratings.
- **Product links** — Open the original product listing.
- **Interactive interface** — Explore recommendations through a Streamlit web app.

### User Flow
1. Search for a product by name.
2. View the selected product.
3. Explore similar product recommendations.
4. Compare products and visit product links.

### Tech Stack
Python, Pandas, Scikit-learn, Streamlit

**Dataset:** Amazon product dataset from Kaggle. Place `amazon.csv` in `data/raw/` before running the app.

## 🧠 How it works

Findly uses **content-based filtering** to recommend products with similar descriptions and categories.

```text
Amazon Product Dataset
          ↓
   Data Preparation
          ↓
 Combine Product Text
          ↓
       TF-IDF
          ↓
  Cosine Similarity
          ↓
 Similar Product Ranking
          ↓
    Findly Web App
```

**Under the hood:**

- **Pandas** handles dataset loading and cleaning.
- **TF-IDF** converts product text into numerical features.
- **Cosine similarity** measures how similar product descriptions are.
- **Streamlit** powers the interactive interface.

## 🚀 Run it locally

### 1. Clone the repository

```bash
git clone https://github.com/Shreyaa712/product-recommendation-system.git
cd product-recommendation-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the Amazon product dataset from Kaggle and place the CSV at:

```text
data/raw/amazon.csv
```

The dataset is not included in this repository.

### 5. Launch Findly

```bash
python -m streamlit run app.py
```

The app will open in your browser.

## 📂 Project structure

```text
product-recommendation-system/
├── app.py
├── recommend.py
├── requirements.txt
├── data/
│   └── raw/
│       └── amazon.csv
├── .gitignore
└── README.md
```

*The dataset is stored locally and excluded from Git.*

##  What's next?

- Add price and rating filters.
- Improve search and product selection.
- Experiment with alternative recommendation methods.
- Explore ways to evaluate recommendation quality.

---

**Built as a hands-on project to learn recommendation systems, text processing, and machine learning in Python.** 💜
