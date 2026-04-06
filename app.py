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

# --- CONFIG ---
st.set_page_config(page_title="GeoSum", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Add this line here to "trick" Google into thinking you are a browser
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

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
    # 1. PERSONA INJECTION: We wrap the headlines in a command
    # This forces BART to 'react' to the text rather than just copy it.
    instruction_phrase = "Provide a cohesive, professional summary of these regional environmental developments:"
    prompt = f"{instruction_phrase} {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # 2. HYPER-PARAMETER TUNING
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_new_tokens=100, 
        min_length=30,
        num_beams=8,               # Higher beams for better sentence structure
        do_sample=True, 
        temperature=1.5,           # High temperature for maximum rephrasing
        top_k=50,
        top_p=0.9,
        repetition_penalty=5.0,    # MAXIMUM penalty to stop it from copying exact words
        no_repeat_ngram_size=3,    # Forbids repeating any 3-word sequence
        early_stopping=True
    )
    
    decoded_summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
    
    # 3. CLEANING & POST-PROCESSING
    # Force string conversion and remove slashes
    clean_summary = str(decoded_summary).replace('\\', '').strip()
    
    # CRITICAL: Remove the instruction phrase if the AI repeated it in the output
    clean_summary = clean_summary.replace(instruction_phrase, "").strip()
    
    # Final check: If it still looks too much like a list, we add a manual lead-in
    if any(clean_summary.lower().startswith(word) for word in ["the", "this", "regional", "analysis"]):
        return clean_summary
    else:
        return f"Analysis of recent regional environmental reports indicates that {clean_summary}"

from gnews import GNews

def fetch_news(loc_name, target_date):
    # Initialize GNews with specific settings
    google_news = GNews(language='en', country='IN', period='7d', max_results=4)
    
    # Create the search query
    query = f"{loc_name} environmental climate news"
    
    try:
        # GNews handles the heavy lifting of fetching and parsing
        results = google_news.get_news(query)
        articles = []
        
        for item in results:
            articles.append({
                'title': item['title'].rsplit(" - ", 1),
                'source': item['publisher']['title'],
                'link': item['url']
            })
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
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
        with st.spinner(f"Analyzing news..."):
            location = geolocator.reverse(f"{st.session_state.lat}, {st.session_state.lon}", language='en')
            loc_name = location.raw.get('address', {}).get('state', search) if location else search

            news = fetch_news(loc_name, selected_date)

            if news:
                # Joining with proper sentence structure for the AI
                headlines = " ".join([f"{n['title']}." for n in news])

                summary = summarize_text(headlines)

                # --- NEW CLEANUP LOGIC ---
                # This removes those slashes (\) and ensures it stays a clean string
                final_summary = str(summary).replace('\\', '').replace("[", "").replace("]", "").replace("'", "").replace('"', "")

                st.success(f"Report for: {loc_name} ({selected_date})")
                
                # --- NEW LOGIC FOR SINGLE NEWS OR REPETITION ---
                if len(news) == 1:
                    st.info(f"**AI Insight:** Recent reports from the region indicate: {final_summary} For more detailed context, please refer to the source link below.")
                else:
                    st.info(f"**AI Insight:** {final_summary}")
                
                st.caption("Click the headlines below to visit the official news pages for more information.")

                for n in news:
                    st.markdown(f"- [{n['title']}]({n['link']}) *({n['source']})*")
            else:
                st.warning(f"No news found for this area on {selected_date}.")
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
