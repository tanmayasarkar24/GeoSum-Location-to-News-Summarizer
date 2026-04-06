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

# --- GEOLOCATOR SETUP ---
# Updated with a unique user_agent and a timeout to prevent connection errors
from geopy.extra.rate_limiter import RateLimiter # Add this import at the top

# --- GEOLOCATOR SETUP ---
geolocator = Nominatim(
    user_agent="tanmaya_sarkar_geosum_final_v3", # Change the name again to 'reset' your ID
    timeout=10
)

# Add this line to create a 'patient' version of the geolocator
reverse_gate = RateLimiter(geolocator.reverse, min_delay_seconds=1.1)
geocode_gate = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

# --- SESSION STATE ---
if "lat" not in st.session_state: st.session_state.lat = 12.823
if "lon" not in st.session_state: st.session_state.lon = 80.044

# --- FUNCTIONS ---

def summarize_text(text):
    instruction = "Provide a cohesive, professional summary of these regional environmental developments:"
    clean_input = str(text).replace("[", "").replace("]", "").replace("'", "")
    prompt = f"{instruction} {clean_input}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_new_tokens=100, 
        min_length=35,
        num_beams=10,               
        do_sample=True, 
        temperature=1.3,           
        repetition_penalty=10.0,    
        no_repeat_ngram_size=3,    
        early_stopping=True
    )
    
    decoded_summary = str(tokenizer.decode(summary_ids, skip_special_tokens=True))
    clean_summary = re.sub(re.escape(instruction), '', decoded_summary, flags=re.IGNORECASE).strip()
    clean_summary = clean_summary.replace('\\', '').replace('"', '').strip()
    
    if clean_summary.lower().startswith("analysis"):
        return clean_summary
    else:
        return f"Analysis of recent regional environmental reports indicates that {clean_summary}"

def fetch_news(loc_name, target_date):
    # 1. SETUP DATE RANGE: GNews works best with a start and end date
    # We look for news from the target date and the day before it
    start_date = target_date - timedelta(days=1)
    
    # 2. INITIALIZE GNEWS WITHOUT A FIXED PERIOD
    # Setting period=None allows the start_date and end_date to take effect
    google_news = GNews(language='en', country='IN', max_results=4)
    google_news.start_date = (start_date.year, start_date.month, start_date.day)
    google_news.end_date = (target_date.year, target_date.month, target_date.day)
    
    # 3. SPECIFIC QUERY
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
        st.error(f"Error fetching news: {e}")
        return []

# --- UI ---
st.title("🌍 GeoSum: Location-to-News Summarizer")

col_loc, col_map = st.columns([1, 1.2])

with col_loc:
    st.subheader("Select Parameters")
    search = st.text_input("🔍 Search Region", value="Tamil Nadu")
    if st.button("Update Region"):
        loc = geocode_gate(search)
        if loc:
            st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
            st.rerun()

    # The date selected here now correctly filters the news
    selected_date = st.date_input("📅 Select Date for News", datetime.now())

    st.markdown("---")

    if st.button("🚀 Generate AI Report", use_container_width=True):
        with st.spinner(f"Searching archives for {selected_date}..."):
            location = reverse_gate(f"{st.session_state.lat}, {st.session_state.lon}", language='en')
            loc_name = location.raw.get('address', {}).get('state', search) if location else search

            news = fetch_news(loc_name, selected_date)

            if news:
                headlines_text = " ".join([f"{n['title']}." for n in news])
                summary = summarize_text(headlines_text)
                final_summary = str(summary).replace("[", "").replace("]", "").replace("'", "").replace('"', "")

                st.success(f"Report for: {loc_name} (Results up to {selected_date})")
                st.info(f"**AI Insight:** {final_summary}")
                
                st.caption("Sources utilized for this analysis:")
                for n in news:
                    st.markdown(f"- [{n['title']}]({n['link']}) *({n['source']})*")
            else:
                st.warning(f"No specific environmental news found for {loc_name} on {selected_date}. Try an earlier date.")

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
