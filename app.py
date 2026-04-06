import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from geopy.geocoders import Nominatim
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime, timedelta
import re
from gnews import GNews

# --- CONFIG ---
st.set_page_config(page_title="GeoSum", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_nlp_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

tokenizer, model = load_nlp_model()
geolocator = Nominatim(user_agent="geosum_final_v1")

# --- SESSION STATE ---
if "lat" not in st.session_state: st.session_state.lat = 12.823
if "lon" not in st.session_state: st.session_state.lon = 80.044

# --- FUNCTIONS ---

def summarize_text(text):
    # 1. We use a 'marker' to separate instructions from data
    # This helps the model understand that the text after 'SUMMARY:' is the goal
    prompt = f"Summarize these environmental news headlines into one paragraph without copying titles: {text} . SUMMARY:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # 2. Stronger generation constraints
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_new_tokens=100, 
        min_length=30,
        num_beams=8,               
        do_sample=True, 
        temperature=1.1,           
        repetition_penalty=12.0,    # High penalty to stop word-for-word copying
        no_repeat_ngram_size=3,     # Forbids repeating 3-word sequences
        early_stopping=True
    )
    
    decoded = str(tokenizer.decode(summary_ids, skip_special_tokens=True))
    
    # 3. THE FIX: We split the output by our marker 'SUMMARY:' 
    # This discards everything before the marker (the leaked instructions)
    if "SUMMARY:" in decoded:
        clean_summary = decoded.split("SUMMARY:")[-1].strip()
    else:
        clean_summary = decoded.strip()

    # 4. Final cleaning of brackets and technical noise
    # We use regex to remove anything that looks like [ 'Title', 'Source' ]
    clean_summary = re.sub(r"\[|\]|'|\"|\\", "", clean_summary)
    
    # Remove common 'leaked' instruction phrases manually just in case
    ban_phrases = [
        "Summarize the following", "Do not list", "cohesive professional summary", 
        "regional environmental developments", "Analysis of recent"
    ]
    for phrase in ban_phrases:
        clean_summary = re.sub(phrase, "", clean_summary, flags=re.IGNORECASE)

    # 5. Professional Prefix
    return f"Analysis of recent regional environmental reports indicates that {clean_summary.strip()}"

def fetch_news(loc_name, target_date):
    start_date = target_date - timedelta(days=2) # Slightly wider window for better results
    google_news = GNews(language='en', country='IN', max_results=4)
    google_news.start_date = (start_date.year, start_date.month, start_date.day)
    google_news.end_date = (target_date.year, target_date.month, target_date.day)
    
    query = f"{loc_name} environmental climate"
    
    try:
        results = google_news.get_news(query)
        articles = []
        for item in results:
            full_title = item.get('title', 'News')
            display_title = full_title.rsplit(" - ", 1) if " - " in full_title else full_title
            articles.append({
                'title': display_title,
                'source': item['publisher']['title'],
                'link': item['url']
            })
        return articles
    except Exception as e:
        return []

# --- UI ---
st.title("🌍 GeoSum: Location-to-News Summarizer")

col_loc, col_map = st.columns([1, 1.2])

with col_loc:
    st.subheader("Select Parameters")
    search = st.text_input("🔍 Search Region", value="Tamil Nadu")
    if st.button("Update Region"):
        loc = geolocator.geocode(search)
        if loc:
            st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
            st.rerun()

    selected_date = st.date_input("📅 Select Date for News", datetime.now())
    st.markdown("---")

    if st.button("🚀 Generate AI Report", use_container_width=True):
        with st.spinner(f"Synthesizing environmental intelligence..."):
            location = geolocator.reverse(f"{st.session_state.lat}, {st.session_state.lon}", language='en')
            loc_name = location.raw.get('address', {}).get('state', search) if location else search

            news = fetch_news(loc_name, selected_date)

            if news:
                headlines_text = " ".join([f"{n['title']}." for n in news])
                summary = summarize_text(headlines_text)

                st.success(f"Report for: {loc_name} (Updated to {selected_date})")
                
                # Using the AI Insight label you requested
                st.info(f"**AI Insight:** {summary}")
                
                st.caption("Sources utilized for this intelligence report:")
                for n in news:
                    st.markdown(f"- [{n['title']}]({n['link']}) *({n['source']})*")
            else:
                st.warning(f"No specific environmental reports found for {loc_name} around {selected_date}.")

with col_map:
    st.write("### Choose Your Location")
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=6)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}',
        attr='Google', name='Satellite Labels', overlay=False
    ).add_to(m)

    Draw(
        export=False,
        draw_options={'polyline': False, 'circle': False, 'circlemarker': False, 'marker': False, 'rectangle': True, 'polygon': False},
        edit_options={'edit': False, 'remove': False}
    ).add_to(m)

    output = st_folium(m, height=500, width=800, key="eco_map")

    if output and output.get("last_clicked"):
        st.session_state.lat = output["last_clicked"]["lat"]
        st.session_state.lon = output["last_clicked"]["lng"]
        st.rerun()
