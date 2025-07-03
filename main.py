import streamlit as st
import requests
import json
import pandas as pd
import io
import os
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- Google Custom Search function ---
def google_custom_search(query, api_key, cx):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {"error": f"Google Custom Search API error: {response.status_code} - {response.text}"}
    data = response.json()
    return data.get("items", [])

# --- Serper Shopping API function ---
def search_serper_shopping(query: str, country: str = "us"):
    url = "https://google.serper.dev/shopping"
    payload = json.dumps({
        "q": query,
        "gl": country.lower()
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        return {"error": f"Serper API error: {response.status_code} - {response.text}", "results": []}
    shopping_results = response.json().get("shopping", [])
    shopping_results = shopping_results[:41]  # limit to 41 results
    return {"error": None, "results": shopping_results}

def format_results_for_csv(product_title, results, error=None):
    row = {"Product Title": product_title, "Search Status": "Error" if error else "Success"}
    if error:
        row["Error Message"] = error
        return row

    for i, r in enumerate(results, start=1):
        row[f"Title {i}"] = r.get("title", "")
        row[f"Source {i}"] = r.get("source", "")
        row[f"Link {i}"] = r.get("link", "")
        row[f"Price {i}"] = r.get("price", "")
        row[f"ImageURL {i}"] = r.get("imageUrl", "")
        row[f"Rating {i}"] = r.get("rating", "")
        row[f"RatingCount {i}"] = r.get("ratingCount", "")
        row[f"ProductId {i}"] = r.get("productId", "")
        row[f"Position {i}"] = r.get("position", "")
    return row

# --- Streamlit UI ---

st.title("🔍 Product Info App")

mode = st.sidebar.selectbox("Select Search Mode", ["Google Custom Search", "Serper Shopping"])

if mode == "Google Custom Search":
    st.header("Google Custom Search for Products / ASIN")

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        st.error("Google API key or Custom Search Engine ID not found in environment variables.")
    else:
        query = st.text_input("Enter product name or ASIN (e.g. B08VF4SCD1)")

        if st.button("Search"):
            if not query:
                st.warning("Please enter a product name or ASIN to search.")
            else:
                with st.spinner("Searching Google Custom Search..."):
                    results = google_custom_search(query, GOOGLE_API_KEY, GOOGLE_CSE_ID)
                    if isinstance(results, dict) and results.get("error"):
                        st.error(results["error"])
                    elif not results:
                        st.info("No results found.")
                    else:
                        st.success(f"Found {len(results)} results:")
                        for i, item in enumerate(results, 1):
                            st.markdown(f"**{i}. [{item.get('title', '')}]({item.get('link', '')})**")
                            st.write(item.get('snippet', ''))
                            st.markdown("---")

elif mode == "Serper Shopping":
    st.header("Serper Shopping Search")

    if not SERPER_API_KEY:
        st.error("Serper API key not found in environment variables.")
    else:
        serper_mode = st.radio("Select Serper Search Mode", ["Search by Name", "Bulk Upload CSV"])

        if serper_mode == "Search by Name":
            product_name = st.text_input("Enter product name to search")

            country_code = st.text_input("Enter country code (default: us)", value="us")

            if st.button("Search"):
                if not product_name:
                    st.warning("Please enter a product name to search.")
                else:
                    with st.spinner(f"Searching Serper for '{product_name}'..."):
                        response = search_serper_shopping(product_name, country_code)
                        if response["error"]:
                            st.error(response["error"])
                        else:
                            results = response["results"]
                            st.success(f"Found {len(results)} results:")
                            for i, r in enumerate(results, 1):
                                st.markdown(f"**{i}. [{r.get('title','')}]({r.get('link','')})**")
                                st.write(f"Source: {r.get('source','')}")
                                st.write(f"Price: {r.get('price','')}")
                                st.write(f"Rating: {r.get('rating','')} ({r.get('ratingCount','')} reviews)")
                                if r.get('imageUrl'):
                                    st.image(r.get('imageUrl'), width=120)
                                st.markdown("---")

        elif serper_mode == "Bulk Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with 'Product Title' column and optional 'Country'", type=["csv"])

            if uploaded_file:
                input_df = pd.read_csv(uploaded_file)

                if "Product Title" not in input_df.columns:
                    st.error("CSV must have a 'Product Title' column.")
                    st.stop()

                if "Country" not in input_df.columns:
                    input_df["Country"] = "us"
                else:
                    input_df["Country"] = input_df["Country"].fillna("us")

                if st.button("Run Bulk Search"):
                    all_rows = []
                    for idx, row in input_df.iterrows():
                        product_title = str(row["Product Title"]).strip()
                        country = str(row["Country"]).strip().lower()

                        if not product_title:
                            continue

                        with st.spinner(f"Searching for: {product_title}"):
                            response = search_serper_shopping(product_title, country)
                            if response["error"]:
                                formatted_row = format_results_for_csv(product_title, [], error=response["error"])
                            else:
                                formatted_row = format_results_for_csv(product_title, response["results"])
                            all_rows.append(formatted_row)

                    if all_rows:
                        result_df = pd.DataFrame(all_rows)

                        st.success("Bulk search completed.")

                        # Create CSV in-memory buffer for download
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Bulk Search Results CSV",
                            data=csv_buffer.getvalue().encode("utf-8"),
                            file_name="serper_bulk_search_results.csv",
                            mime="text/csv"
                        )

