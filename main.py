import streamlit as st
import requests
import json
import pandas as pd
import io
import os
from dotenv import load_dotenv
from groq import Groq
import tiktoken

# Load API keys
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Serper Shopping API ---
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
    shopping_results = shopping_results[:41]
    return {"error": None, "results": shopping_results}

def format_results_for_csv(product_title, model_number, results, error=None):
    rows = []
    if error:
        return [{"Product Title": product_title, "Model Number": model_number, "Error": error}]
    for r in results:
        rows.append({
            "Product Title": product_title,
            "Model Number": model_number,
            "Title": r.get("title", ""),
            "Source": r.get("source", ""),
            "Link": r.get("link", ""),
            "Price": r.get("price", ""),
            "Rating": r.get("rating", ""),
            "RatingCount": r.get("ratingCount", ""),
            "ImageURL": r.get("imageUrl", "")
        })
    return rows

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def get_links_matching_identifier(data: list, identifier: str):
    prompt = f"""
You are a filtering assistant. Given the following product search results, extract only the links that clearly match the product identifier: "{identifier}".

Each item has: title, source, link.

Return your answer ONLY as a JSON array in this format:
[
  {{"title": "...", "source": "...", "link": "..." }},
  ...
]

Do NOT include any explanation. Just the JSON array.

Input:
{json.dumps(data[:20], indent=2)}
"""
    tokens_used = count_tokens(prompt, model_name="llama3-70b-8192")
    st.write(f"⚡ Estimated tokens in prompt: {tokens_used}")

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw_response = response.choices[0].message.content.strip()

    st.markdown("### 🧪 Raw LLM Response")
    st.code(raw_response, language="json")

    try:
        matches = json.loads(raw_response)
    except Exception as e:
        st.error(f"❌ Failed to parse LLM response: {e}")
        matches = []

    return matches

# --- Streamlit UI ---
st.set_page_config(page_title="🛒 Product Data Fetching and Filtering", layout="wide")
st.title("🛍️ Product Data Fetching and Filtering with Model Number")

tabs = st.tabs(["🔎 Search Single Product", "📂 Bulk Upload & Filter"])

# --- Tab 1: Single Product Search ---
with tabs[0]:
    st.subheader("🔎 Search Product by Name and Model Number")

    product_name = st.text_input("Enter product name", placeholder="e.g., BenQ DesignVue PD2705Q")
    model_number = st.text_input("Enter model number", placeholder="e.g., PD2705Q")
    country_code = st.text_input("Country Code (default: us)", value="us")

    if st.button("Fetch and Filter Results"):
        if not product_name or not model_number:
            st.warning("Please enter both product name and model number.")
        else:
            with st.spinner("Searching Serper..."):
                result = search_serper_shopping(product_name, country_code)
                if result["error"]:
                    st.error(result["error"])
                else:
                    full_rows = format_results_for_csv(product_name, model_number, result["results"])
                    full_df = pd.DataFrame(full_rows)

                    st.markdown("### 🧾 All Fetched Results")
                    st.dataframe(full_df)

                    # Download all results
                    full_buffer = io.StringIO()
                    full_df.to_csv(full_buffer, index=False)
                    st.download_button(
                        label="📥 Download All Fetched Results CSV",
                        data=full_buffer.getvalue().encode("utf-8"),
                        file_name="all_fetched_results.csv",
                        mime="text/csv"
                    )

                    # Filter with LLM
                    matches = get_links_matching_identifier(
                        full_df[["Title", "Source", "Link"]].to_dict(orient="records"),
                        model_number
                    )
                    filtered_df = pd.DataFrame()
                    if matches:
                        filtered_df = pd.merge(
                            pd.DataFrame(matches),
                            full_df,
                            left_on=["title", "source", "link"],
                            right_on=["Title", "Source", "Link"],
                            how="inner"
                        )[full_df.columns]

                        st.success(f"LLM filtered {len(filtered_df)} relevant links.")
                        st.markdown("### ✅ Filtered Results")
                        st.dataframe(filtered_df)

                        out_buffer = io.StringIO()
                        filtered_df.to_csv(out_buffer, index=False)
                        st.download_button(
                            label="📥 Download Filtered Links CSV",
                            data=out_buffer.getvalue().encode("utf-8"),
                            file_name="filtered_links_single_product.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No matching links found.")

# --- Tab 2: Bulk Upload ---
with tabs[1]:
    st.subheader("📂 Upload CSV with columns: Product Title, Model Number, Country Code")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if st.button("Run Bulk Fetch and Filter"):
        if not uploaded_file:
            st.warning("Please upload a CSV file.")
        else:
            df_input = pd.read_csv(uploaded_file)
            required_cols = ["Product Title", "Model Number", "Country Code"]
            if not all(col in df_input.columns for col in required_cols):
                st.error(f"CSV must include these columns: {required_cols}")
            else:
                all_full_rows = []
                all_filtered_rows = []
                total = len(df_input)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, row in df_input.iterrows():
                    product_title = str(row["Product Title"]).strip()
                    model_number = str(row["Model Number"]).strip()
                    country_code = str(row["Country Code"]).strip() or "us"

                    status_text.text(f"Processing {i+1}/{total}: {product_title} ({model_number})")

                    query = f"{product_title} {model_number}"
                    result = search_serper_shopping(query, country_code)
                    full_rows = format_results_for_csv(product_title, model_number, result["results"], result["error"])
                    all_full_rows.extend(full_rows)

                    # Filter results using LLM
                    data_to_filter = [{"title": r["Title"], "source": r["Source"], "link": r["Link"]}
                                      for r in full_rows if "Error" not in r]
                    filtered = get_links_matching_identifier(data_to_filter, model_number)
                    if filtered:
                        full_df = pd.DataFrame(full_rows)
                        filtered_df = pd.merge(
                            pd.DataFrame(filtered),
                            full_df,
                            left_on=["title", "source", "link"],
                            right_on=["Title", "Source", "Link"],
                            how="inner"
                        )[full_df.columns]
                        all_filtered_rows.extend(filtered_df.to_dict(orient="records"))
                    else:
                        all_filtered_rows.extend(full_rows)  # fallback

                    progress_bar.progress((i + 1) / total)

                progress_bar.empty()
                status_text.text("✅ Bulk fetch and filtering completed.")

                df_all = pd.DataFrame(all_full_rows)
                st.markdown("### 📊 All Fetched Results")
                st.dataframe(df_all)
                full_csv = io.StringIO()
                df_all.to_csv(full_csv, index=False)
                st.download_button("📥 Download All Fetched Results CSV", full_csv.getvalue().encode(), "bulk_all_results.csv", mime="text/csv")

                df_filtered = pd.DataFrame(all_filtered_rows)
                st.markdown("### ✅ Filtered Results")
                st.dataframe(df_filtered)
                filtered_csv = io.StringIO()
                df_filtered.to_csv(filtered_csv, index=False)
                st.download_button("📥 Download Filtered Results CSV", filtered_csv.getvalue().encode(), "bulk_filtered_results.csv", mime="text/csv")


# --- Tab 3: AI Generated Columns (Expander) ---
with st.expander("🤖 Append AI-Generated Data to Filtered CSV"):
    st.markdown("Upload the filtered CSV and another CSV specifying products that need AI-generated fields (Category, Description, Keywords, MetaTitle).")

    col1, col2 = st.columns(2)
    with col1:
        filtered_csv_file = st.file_uploader("Upload Filtered CSV (from earlier)", type=["csv"], key="filtered_csv")
    with col2:
        ai_template_file = st.file_uploader("Upload CSV with AI Fields to Generate", type=["csv"], key="ai_template_csv")

    if st.button("✨ Generate AI Columns and Append to Filtered CSV"):
        if not filtered_csv_file or not ai_template_file:
            st.warning("Please upload both CSV files.")
        else:
            filtered_df = pd.read_csv(filtered_csv_file)
            ai_template_df = pd.read_csv(ai_template_file)

            if "Model Number" not in filtered_df.columns or "Model Number" not in ai_template_df.columns:
                st.error("Both CSVs must have a 'Model Number' column.")
            else:
                merged_df = filtered_df.merge(ai_template_df, on="Model Number", how="left", suffixes=("", "_ai"))

                new_category = []
                new_description = []
                new_keywords = []
                new_meta_title = []

                prompt_template = """
You are an AI assistant that generates product metadata.

Given product details:

Product Title: {title}
Model Number: {model_number}

Generate:

1. Category (short phrase)
2. SEO-friendly Technical Description in 5 lines (in bullet points)
3. Keywords (comma-separated)
4. Meta Title (SEO optimized)

Return the answer as JSON with keys: Category, Description, Keywords, MetaTitle.
"""

                for idx, row in merged_df.iterrows():
                    # If AI fields already filled, reuse
                    if pd.notna(row.get("Category")) and pd.notna(row.get("Description")) and pd.notna(row.get("Keywords")) and pd.notna(row.get("MetaTitle")):
                        new_category.append(row["Category"])
                        new_description.append(row["Description"])
                        new_keywords.append(row["Keywords"])
                        new_meta_title.append(row["MetaTitle"])
                        continue

                    title = row.get("Title") or row.get("Product Title") or ""
                    model_num = row.get("Model Number") or ""

                    prompt = prompt_template.format(title=title, model_number=model_num)

                    try:
                        response = groq_client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        content = response.choices[0].message.content.strip()
                        # Parse JSON from response (some LLMs return text with explanation, so attempt best effort)
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        data_json = json.loads(json_str)

                        new_category.append(data_json.get("Category", ""))
                        new_description.append(data_json.get("Description", ""))
                        new_keywords.append(data_json.get("Keywords", ""))
                        new_meta_title.append(data_json.get("MetaTitle", ""))
                    except Exception as e:
                        st.warning(f"AI generation failed for row {idx} ({model_num}): {e}")
                        new_category.append("")
                        new_description.append("")
                        new_keywords.append("")
                        new_meta_title.append("")

                merged_df["Category"] = new_category
                merged_df["Description"] = new_description
                merged_df["Keywords"] = new_keywords
                merged_df["MetaTitle"] = new_meta_title

                st.markdown("### 📄 AI-Enhanced Data Preview")
                st.dataframe(merged_df.head(10))

                csv_buffer = io.StringIO()
                merged_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download AI-Enhanced CSV",
                    data=csv_buffer.getvalue().encode("utf-8"),
                    file_name="ai_enhanced_filtered_data.csv",
                    mime="text/csv"
                )


st.sidebar.title("Complete AI Metadata Generation Guide")

help_text = """
### Step 1: Upload Filtered CSV File  
- Click the **Upload Filtered CSV** button.  
- Select your main dataset CSV file containing products with necessary columns like **Product Title** and **Model Number**.  
- This is the base data for AI to work on.

### Step 2: Download Filtered CSV (Optional)  
- After uploading, you can download this filtered CSV for backup or review by clicking the **Download Filtered CSV** button.

### Step 3: Upload AI Column CSV File  
- Click the **Upload AI Column CSV** button.  
- Upload a CSV file containing the list of products or columns you want the AI to generate metadata for.  
- This file must contain columns like **Product Title** and **Model Number** matching your filtered CSV.

### Step 4: Generate AI Metadata  
- Click the **Generate AI Metadata** button.  
- The AI will analyze the uploaded data and generate new columns such as **Category**, **Description**, **Keywords**, and **Meta Titles**.

### Step 5: Preview Results  
- After processing, a preview of the combined dataset with the AI-generated metadata will appear.  
- Review the generated metadata to ensure it meets your expectations.

### Step 6: Download Final CSV  
- Click the **Download Final CSV** button to save the new enriched CSV file to your device.

---


"""

st.sidebar.markdown(help_text)

