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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    # Increased max_new_tokens to ensure all news items are captured in summary
    summary_ids = model.generate(inputs["input_ids"], max_new_tokens=200, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids, skip_special_tokens=True)

def fetch_news(loc_name, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    next_day = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    query = f"{loc_name} environmental climate news after:{date_str} before:{next_day}"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'xml')
        articles = []
        for item in soup.find_all('item', limit=4):
            full_title = str(item.title.text)

            # --- CRITICAL FIX: Add here to get the STRING, not a LIST ---
            title_part = full_title.rsplit(" - ", 1) if " - " in full_title else full_title
            source_part = full_title.rsplit(" - ", 1)[-1] if " - " in full_title else "News"

            articles.append({
                'title': str(title_part).strip(), # Forces it to stay a clean string
                'source': str(source_part).strip(),
                'link': item.link.text
            })
        return articles
    except: return []

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

                # Final cleanup of any potential artifacts
                final_summary = str(summary).replace("[", "").replace("]", "").replace("'", "").replace('"', "")

                st.success(f"Report for: {loc_name} ({selected_date})")
                st.info(f"**Summarized environmental news:** {final_summary}")

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
