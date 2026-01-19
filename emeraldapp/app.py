import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import baseline_run
import em_rag_run
import kg_rag_run
import hybrid_run
import os
import yaml
from dotenv import load_dotenv
from visualize_graph import visualize_evidence_subgraph
from parser import GenericParser
import tempfile
from pathlib import Path
import json
import time
import io
import base64
import warnings
warnings.filterwarnings("ignore", message=".*widget with key.*Session State API.*")
load_dotenv()
DB_PATH = os.getenv("STREAMLIT_CHROMA_DB_PATH")
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
COMPANIES_YAML_PATH = Path(__file__).parent / "utils" / "companies.yaml"

ENABLE_SESSION_CLEANUP = False

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(layout="wide", page_title="EmeraldApp", page_icon="🌿")

if "added_files" not in st.session_state: st.session_state.added_files = []
if "session_id" not in st.session_state: st.session_state.session_id = str(time.time())
if "file_uploader_key" not in st.session_state: st.session_state.file_uploader_key = 0

# --- STATE INITIALIZATION ---
if "few_shot" not in st.session_state: st.session_state.few_shot = True
if "fs_widget" not in st.session_state: st.session_state.fs_widget = True

# ============================================
# HELPER FUNCTIONS
# ============================================

# MOVED UP: Image helper needed for Header
def get_img_as_base64(file_path):
    """
    Reads an image file (png, jpg, svg) and returns a base64 string
    for HTML embedding.
    """
    if not os.path.exists(file_path):
        return ""
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    encoded = base64.b64encode(data).decode()
    
    if file_path.endswith(".svg"):
        mime = "image/svg+xml"
    elif file_path.endswith(".png"):
        mime = "image/png"
    else:
        mime = "image/jpeg"
        
    return f"data:{mime};base64,{encoded}"

def load_companies_and_years():
    if not COMPANIES_YAML_PATH.exists(): return [], {}, {}
    try:
        with open(COMPANIES_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        comps, c_map, y_map = [], {}, {}
        for item in data.get("companies", []):
            display, key, years = item.get("display"), item.get("key"), item.get("years", [])
            if display and key:
                comps.append(display)
                c_map[display] = key
                y_map[display] = sorted([str(y) for y in years], reverse=True)
        return sorted(comps), c_map, y_map
    except Exception: return [], {}, {}

def update_company_in_yaml(company_name, year):
    if not COMPANIES_YAML_PATH.exists():
        data = {"companies": []}
    else:
        try:
            with open(COMPANIES_YAML_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {"companies": []}
        except Exception:
            data = {"companies": []}

    display_name = company_name.replace("_", " ") 
    clean_key = company_name.lower().replace(" ", "_").strip()
    year_str = str(year)
    
    found = False
    if "companies" not in data: data["companies"] = []

    for entry in data["companies"]:
        if entry.get("key") == clean_key or entry.get("display", "").lower() == display_name.lower():
            found = True
            current_years = [str(y) for y in entry.get("years", [])]
            if year_str not in current_years:
                current_years.append(year_str)
                valid_years = [int(y) for y in current_years if y.isdigit()]
                entry["years"] = sorted(valid_years, reverse=True)
            break
    
    if not found:
        new_entry = {
            "display": display_name,
            "key": clean_key,
            "years": [int(year_str)] if year_str.isdigit() else []
        }
        data["companies"].append(new_entry)
        
    data["companies"].sort(key=lambda x: x["display"])
    
    try:
        with open(COMPANIES_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"Error saving YAML: {e}")
        return False

COMPANY_DISPLAY_NAMES, COMPANY_KEY_MAP, COMPANY_YEARS_MAP = load_companies_and_years()

def get_parser():
    def parse_company_filename(filename: str):
        base_name = filename.rsplit(".", 1)[0]
        parts = base_name.split("_")
        if len(parts) >= 2:
            return {"company": "_".join(parts[:-1]), "year": parts[-1], "filename": filename}
        return {"filename": filename}
    return GenericParser(documents_folder="./temp", db_path=DB_PATH, collection_name="company_reports", file_extensions=[".pdf"], metadata_extractor=parse_company_filename)

def save_run_to_history(data):
    try:
        timestamp = int(time.time())
        file_path = HISTORY_DIR / f"run_{timestamp}.json"
        data["timestamp"] = timestamp
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e: st.error(f"Failed to save history: {e}")

def load_all_history():
    history_items = []
    if not HISTORY_DIR.exists(): return []
    for file_path in HISTORY_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_file_path"] = str(file_path)
                history_items.append(data)
        except Exception: continue
    history_items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return history_items

def populate_form_callback():
    selection = st.session_state.get("history_table", {}).get("selection", {})
    if selection and selection.get("rows"):
        idx = selection["rows"][0]
        
        # 1. Load all data
        hist_items = load_all_history()
        
        # 2. Reconstruct the DataFrame to match what the user saw
        table_rows = []
        for item in hist_items:
            lbl = item.get("result", {}).get("label", "")
            v_str = "🔴 Greenwashing" if lbl == "greenwashing" else "🟢 Not Greenwashing" if lbl == "not_greenwashing" else "⚪ Insufficient evidence"
            is_few_shot = item.get("few_shot", False)
            prompt_str = "Few-shot" if is_few_shot else "Zero-shot"
            
            table_rows.append({
                "Company": item.get("company_display"),
                "Claim": item.get("claim"),
                "Verdict": v_str,
                "Pipeline": item.get("pipeline", ""),
                "Type": prompt_str,
                "timestamp": item.get("timestamp") # Crucial for ID matching
            })
        
        df = pd.DataFrame(table_rows)

        # 3. Re-apply the search filter using the session state key
        search_query = st.session_state.get("history_search", "")
        if search_query:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
            df = df[mask]
        
        # 4. Map the visible index (idx) to the unique timestamp
        if idx < len(df):
            selected_row = df.iloc[idx]
            selected_ts = selected_row["timestamp"]
            
            # 5. Find the full original item object based on timestamp
            # We search the original hist_items list for the matching timestamp
            item = next((i for i in hist_items if i.get("timestamp") == selected_ts), None)

            if item:
                try:
                    claim_val = item.get("claim", "")
                    comp_val = item.get("company_display", "")
                    pipe_val = item.get("pipeline", "EM-HYBRID")
                    year_val = str(item.get("year", "")) 
                    fs_val = item.get("few_shot", True)

                    st.session_state.claim_input = claim_val
                    st.session_state.company_select = comp_val
                    st.session_state.pipeline = pipe_val
                    st.session_state.year_select = year_val
                    st.session_state.few_shot = fs_val
                    st.session_state.result_data = item

                    st.session_state.widget_claim = claim_val
                    st.session_state.widget_company = comp_val
                    st.session_state.widget_pipeline = pipe_val
                    st.session_state.fs_widget = fs_val
                    st.session_state.widget_year = year_val       
                    st.session_state.widget_year_text = year_val  

                except Exception as e:
                    st.error(f"Error loading history item: {e}")

# ============================================
# CSS STYLING & HEADER
# ============================================

# Load logo specifically for the header
# Assuming 'site_logo.png' is the same logo you want in the header
header_logo_b64 = get_img_as_base64("site_logo.png")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #333;
        background-color: #F5F5F5;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }}

    /* --- GRADIENTS --- */
    .app-header-container, 
    .result-header, 
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #435334 0%, #5a6d42 50%, #9eb384 100%) !important;
        color: white !important;
        border: none !important;
    }}

    /* --- HEADER --- */
    header[data-testid="stHeader"] {{ display: none; }}
    .app-header-container {{
        position: fixed; top: 0; left: 0; width: 100%;
        padding: 12px 32px;
        display: flex; align-items: center; gap: 16px;
        z-index: 99999; height: 74px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }}
    .header-logo-box {{
        width: 44px; height: 44px; background: white;
        border-radius: 10px; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .header-logo-inner {{
        width: 36px; height: 36px;
        border-radius: 6px; display: flex; align-items: center; justify-content: center;
        /* Removed Text Styling since it's an image now */
        overflow: hidden;
    }}
    .header-logo-inner img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        padding: 4px; /* Optional: Adds a little breathing room inside the green box */
    }}

    .header-title {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        display: flex;
        flex-direction: column;
    }}

    .header-title h1 {{ 
        font-size: 1.6rem; 
        font-weight: 700; 
        color: white; 
        margin: 0; 
        line-height: 1; 
        letter-spacing: -0.3px;
    }}

    .header-title p {{ 
        font-size: 0.85rem; 
        color: rgba(255,255,255,0.9); 
        font-weight: 400;
        letter-spacing: -0.2px;
        line-height: 1.2; 
        margin: 0;
        margin-top: 4px; 
    }}

    .block-container {{ padding-top: 100px !important; padding-bottom: 40px; }}

    /* --- SIDEBAR COMPONENTS --- */
    .stTextArea textarea {{
        border: 1px solid #ccc !important; border-radius: 8px !important; background: white !important;
    }}
    .stTextArea textarea:focus {{ border-color: #6B8E6B !important; box-shadow: 0 0 0 1px #6B8E6B !important; }}
    
    .stButton > button[kind="primary"] {{
        color: white !important; border: none; font-weight: 600;
        border-radius: 6px; padding: 0.5rem 1rem;
    }}
    
    /* Toggle switch - green when ON */
    div[data-testid="stToggle"] > label > div[role="checkbox"][aria-checked="true"],
    div[role="switch"][aria-checked="true"] {{
        background-color: #5a6d42 !important;
        border-color: #5a6d42 !important;
    }}
    
    /* Toggle thumb color */
    div[data-testid="stToggle"] > label > div[role="checkbox"] > div {{
        background-color: white !important;
    }}
    
    /* DataFrame selection highlighting - green */
    div[data-testid="stDataFrame"] {{
        --primary-color: #5a6d42 !important;
        font-size: 14px !important;
    }}
    
    /* DataFrame row selection - override blue with green */
    div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
        --gdg-accent-color: #5a6d42 !important;
        --gdg-accent-fg: white !important;
        --gdg-accent-light: rgba(90, 109, 66, 0.15) !important;
    }}
    
    /* Selected row background */
    div[data-testid="stDataFrame"] .dvn-scroller [aria-selected="true"],
    div[data-testid="stDataFrame"] .row-selected {{
        background-color: rgba(90, 109, 66, 0.2) !important;
    }}
    
    section[data-testid="stFileUploaderDropzone"] {{
        background-color: #f9fdf6; border: 1px dashed #6B8E6B;
    }}

    /* --- RESULT CARD CONTAINER --- */
    .result-card {{
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        transform: translateZ(0);
        backface-visibility: hidden;
    }}
    .result-header {{
        color: white; padding: 12px 24px; font-weight: 600; font-size: 1rem;
    }}
    .result-body {{ padding: 24px; }}

    /* Claim Box */
    .claim-wrapper {{ display: flex; gap: 20px; margin-bottom: 25px; }}
    .claim-text-col {{
        flex: 1; background: #fcfcfc; padding: 16px;
        border-left: 4px solid #6B8E6B; border-radius: 4px;
        border: 1px solid #f0f0f0; border-left-width: 4px;
    }}
    .claim-label {{ font-size: 0.75rem; color: #666; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; }}
    .claim-content {{ font-style: italic; font-size: 1.1rem; color: #222; margin-bottom: 12px; }}
    .claim-meta {{ font-size: 0.85rem; color: #888; }}
    
    /* Verdicts */
    .verdict-box {{
        width: 220px; display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1.1rem; border-radius: 8px; padding: 10px; text-align: center;
        height: fit-content; align-self: flex-start;
    }}
    .v-greenwash {{ background: #ffeaea; color: #D64545; border: 2px solid #feb2b2; }}
    .v-clean {{ background: #e6ffed; color: #2F855A; border: 2px solid #9ae6b4; }}
    .v-insufficient {{ background: #fffaf0; color: #9C4221; border: 2px solid #fbd38d; }}

    /* --- HTML DETAILS/SUMMARY --- */
    details.custom-details {{
        background: #fcfcfc; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 12px; overflow: hidden;
    }}
    summary.custom-summary {{
        background: #f4f5f7; padding: 12px 16px; cursor: pointer; font-weight: 600; color: #333;
        list-style: none; display: flex; align-items: center;
    }}
    summary.custom-summary:hover {{ background: #eaebef; }}
    summary.custom-summary::-webkit-details-marker {{ display: none; }}
    summary.custom-summary::before {{
        content: "▶"; font-size: 0.8rem; margin-right: 10px; transition: transform 0.2s; color: #555;
    }}
    details[open] summary.custom-summary::before {{ transform: rotate(90deg); }}
    details[open] summary.custom-summary {{ border-bottom: 1px solid #e0e0e0; }}
    .details-content {{ padding: 16px; font-size: 0.95rem; line-height: 1.5; color: #333; }}

    /* Evidence Cards */
    .html-ev-card {{
        background: white; border: 1px solid #eee; border-radius: 6px;
        padding: 12px; margin-bottom: 10px; border-left: 3px solid #ccc;
        margin-top: 5px;
    }}
    .html-ev-meta {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.8rem; color: #555; font-weight: 600; }}
    .html-score {{ background: #e6ffed; color: #22543d; padding: 1px 8px; border-radius: 4px; }}

    /* Loading */
    .loading-overlay {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.95); z-index: 99999;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }}
    .spinner {{
        border: 4px solid #f3f3f3; border-top: 4px solid #435334;
        border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;
    }}
    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    .justification-scroll {{
        max-height: 250px;       /* Sets the standard fixed height */
        overflow-y: auto;        /* Enables vertical scrolling */
        padding-right: 8px;      /* Prevents text from hitting the scrollbar */
    }}
</style>

<div class="app-header-container">
    <div class="header-logo-box">
        <div class="header-logo-inner">
            <img src="{header_logo_b64}" alt="Logo">
        </div>
    </div>
    <div class="header-title">
        <h1>EmeraldApp</h1>
        <p style="margin-top: -16px; font-size: 0.85em;">AI-Powered ESG Greenwashing Detection</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Loading Overlays
if st.session_state.get("show_loading", False):
    st.markdown("""<div class="loading-overlay"><div class="spinner"></div><div style="margin-top:15px; color:#435334; font-weight:600;">Verifying Claim...</div></div>""", unsafe_allow_html=True)
if st.session_state.get("show_parsing", False):
    st.markdown("""<div class="loading-overlay"><div class="spinner"></div><div style="margin-top:15px; color:#435334; font-weight:600;">Processing Documents...</div></div>""", unsafe_allow_html=True)

# ============================================
# MAIN LAYOUT
# ============================================

col_sidebar, col_main = st.columns([1, 2.8], gap="large")

# --------------------------
# LEFT COLUMN (INPUTS & HISTORY)
# --------------------------
with col_sidebar:
    st.markdown("### Enter Your Claim")
    
    # 1. CLAIM INPUT
    # Initialize state if missing so we don't need 'value=' in the widget
    if "widget_claim" not in st.session_state:
        st.session_state.widget_claim = st.session_state.get("claim_input", "")

    claim = st.text_area(
        "Claim Input", 
        placeholder="Enter the ESG claim you want to verify.", 
        height=120, 
        label_visibility="collapsed", 
        key="widget_claim" 
        # Removed 'value='
    )
    # Sync back to main state
    st.session_state.claim_input = claim

    # 2. COMPANY SELECTION
    # Changed ratio to [0.3, 0.7] to give text more breathing room on small screens
    c_label, c_input = st.columns([0.3, 0.7]) 
    
    with c_label:
        # Removed 'white-space: nowrap' so text wraps instead of overlapping
        # Added 'line-height: 1.2' so if it wraps, it looks neat
        st.markdown('<div style="font-size: 1rem; font-weight: 600; padding-top: 10px; line-height: 1.2;">Select Company</div>', unsafe_allow_html=True)

    with c_input:
        if "widget_company" not in st.session_state:
            default_comp = COMPANY_DISPLAY_NAMES[0] if COMPANY_DISPLAY_NAMES else ""
            st.session_state.widget_company = st.session_state.get("company_select", default_comp)

        company_display = st.selectbox(
            "Company Name", 
            options=COMPANY_DISPLAY_NAMES, 
            key="widget_company", 
            label_visibility="collapsed" 
        )
        
    st.session_state.company_select = company_display

    # 3. ADVANCED SETTINGS (Pipeline, Year, Few-Shot)
    with st.expander("Advanced Settings", expanded=True):
        
        # PIPELINE SELECTION
        pipe_opts = ["EM-HYBRID", "EM-RAG", "EM-KGRAG", "EM-Baseline"]
        
        # Fix: Initialize widget state
        if "widget_pipeline" not in st.session_state:
            st.session_state.widget_pipeline = st.session_state.get("pipeline", "EM-HYBRID")

        pipeline = st.selectbox(
            "Pipeline", 
            options=pipe_opts, 
            key="widget_pipeline"
            # Removed 'index='
        )

        # YEAR SELECTION
        # Retrieve mapped years for the selected company
        raw_years = COMPANY_YEARS_MAP.get(company_display, [])
        # Prepend "None" so it is always the first (default) option
        avail_years = ["None"] + raw_years
        
        # Fix: Initialize widget state (Year is tricky because options change based on company)
        # We try to keep the old year if it's valid for the new company, otherwise default to "None" (index 0).
        saved_year = str(st.session_state.get("year_select", ""))
        
        if avail_years:
            if "widget_year" not in st.session_state:
                # If saved year is valid for this company, use it. Else use the first available option ("None").
                initial_year = saved_year if saved_year in avail_years else avail_years[0]
                st.session_state.widget_year = initial_year
            
            # Note: If the user changes Company, 'widget_year' might hold a year not in 'avail_years'.
            # Streamlit handles this gracefully usually, but sometimes resets.
            # Ideally, we force a reset if the value is invalid, but simple removal of 'index' helps most cases.
            
            year = st.selectbox(
                "Year", 
                options=avail_years, 
                key="widget_year"
                # Removed 'index='
            )
        else:
            # Fallback text input (Should rarely be hit if avail_years always has "None")
            if "widget_year_text" not in st.session_state:
                st.session_state.widget_year_text = saved_year
                
            year = st.text_input(
                "Year", 
                key="widget_year_text"
                # Removed 'value='
            )

        st.write("")
        
        # FEW SHOT TOGGLE
        t_col1, t_col2 = st.columns([0.6, 0.4])
        
        curr_state = st.session_state.get("fs_widget", True)
        style_zero = "color: #999; font-weight: 500;" if curr_state else "color: #000; font-weight: 700;"
        style_few = "color: #000; font-weight: 700;" if curr_state else "color: #999; font-weight: 500;"
        
        with t_col1: 
            st.markdown(f"""<div style="text-align:right; padding-top:6px; font-size:0.95rem;"><span style="{style_zero}">Zero-shot</span>&nbsp;&nbsp;<span style="{style_few}">Few-shot</span></div>""", unsafe_allow_html=True)
        with t_col2: 
            def update_fs_logic():
                st.session_state.few_shot = st.session_state.fs_widget
            
            st.toggle("FS", key="fs_widget", on_change=update_fs_logic, label_visibility="collapsed")

    st.write("")
    # 4. ACTION BUTTON
    if st.button("Verify Claim", type="primary", use_container_width=True):
        if not claim: 
            st.warning("Please enter a claim.")
        else:
            # Explicitly sync all values before running to be safe
            st.session_state.claim_input = claim
            st.session_state.pipeline = pipeline
            st.session_state.year_select = year
            st.session_state.show_loading = True
            st.rerun()

    # 5. HISTORY TABLE
    # Wrapped in expander for collapsibility
    with st.expander("Past Claims", expanded=True):
        search_query = st.text_input(
            "Search History", 
            placeholder="🔍 Search company, claim, or verdict...", 
            label_visibility="collapsed",
            key="history_search"
        )

        history = load_all_history()
        if history:
            table_rows = []
            for item in history:
                lbl = item.get("result", {}).get("label", "")
                v_str = "🔴 Greenwashing" if lbl == "greenwashing" else "🟢 Not Greenwashing" if lbl == "not_greenwashing" else "⚪ Insufficient evidence"
                
                is_few_shot = item.get("few_shot", False)
                prompt_str = "Few-shot" if is_few_shot else "Zero-shot"
                pipeline_str = item.get("pipeline", "")
                table_rows.append({
                    "Company": item.get("company_display"),
                    "Claim": item.get("claim"),
                    "Verdict": v_str,
                    "Pipeline": pipeline_str,
                    "Type": prompt_str,
                    "timestamp": item.get("timestamp")
                })
            df = pd.DataFrame(table_rows)

            if search_query:
                mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
                df = df[mask]
            # height=180 shows roughly 4 rows + header. 
            # Streamlit handles the scrolling automatically when content exceeds this height.
            st.dataframe(
                df, 
                key="history_table",
                height=180, 
                column_config={
                    "Company": st.column_config.TextColumn("Company", width="small"),
                    "Claim": st.column_config.TextColumn("Claim", width="small"),
                    # Recommended Change
                    "Verdict": st.column_config.TextColumn("Verdict", width="small"), # or "120px"
                    "Pipeline": st.column_config.TextColumn("Pipeline", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "timestamp": None,
                },
                use_container_width=True, 
                hide_index=True, 
                on_select="rerun", 
                selection_mode="single-row"
            )
            
            if st.button("Populate Form", type="primary", use_container_width=True, on_click=populate_form_callback):
                pass
        else:
            st.info("No history yet.")
    st.markdown("### Upload ESG Report")
    with st.container():
        uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed", key=f"uploader_{st.session_state.file_uploader_key}")
        if uploaded_files:
            if st.button("Process Files", key="btn_proc"):
                st.session_state.uploaded_files_data = uploaded_files; st.session_state.show_parsing = True; st.rerun()

# --------------------------
# RIGHT COLUMN (RESULTS)
# --------------------------
with col_main:
    
    if "result_data" in st.session_state:
        res_data = st.session_state.result_data
        result_obj = res_data.get("result", {})
        
        lbl = result_obj.get("label", "insufficient")
        v_class, v_text = ("v-greenwash", "Greenwashing Detected") if lbl == "greenwashing" else ("v-clean", "Not Greenwashing") if lbl == "not_greenwashing" else ("v-insufficient", "Insufficient Evidence")
        
        justification_text = result_obj.get("justification", "No justification provided.")
        
        # --- EVIDENCE HTML GENERATION ---
        evidence_html = ""
        subgraph_input = result_obj.get("subgraph", None)
        passages = result_obj.get("passages", [])
        
        # Flag to track if we successfully created a graph image
        graph_generated = False

        # 1. TRY SUBGRAPH FIRST
        if subgraph_input:
            try:
                # A. Normalize Input (String -> JSON)
                if isinstance(subgraph_input, str):
                    clean_str = subgraph_input.replace("```json", "").replace("```", "").strip()
                    if clean_str and clean_str not in ["[]", "{}"]:
                        subgraph_input = json.loads(clean_str)
                    else:
                        subgraph_input = None

                # B. Prepare Data for Visualization (ROBUST PARSING)
                flat_evidence = []
                target_company = res_data.get('company_display', 'Company')

                if subgraph_input:
                    # CASE 1: Complex Dictionary with "evidence_by_type" (Original Format)
                    if isinstance(subgraph_input, dict) and "evidence_by_type" in subgraph_input:
                        if "company" in subgraph_input: target_company = subgraph_input["company"]
                        for label, items in subgraph_input["evidence_by_type"].items():
                            for item in items:
                                flat_evidence.append({
                                    "node_id": item.get("id", np.random.randint(1000, 9999)), 
                                    "labels": [label],
                                    "properties": item.get("properties", item), # Fallback if properties aren't nested
                                    "rel_type": item.get("connection", {}).get("relationships", ["RELATED"])[0] if isinstance(item.get("connection"), dict) else "RELATED"
                                })

                    # CASE 2: List of nodes (Simpler Format)
                    elif isinstance(subgraph_input, list):
                        flat_evidence = subgraph_input

                    # CASE 3: NetworkX node-link format (common in Graph RAGs)
                    elif isinstance(subgraph_input, dict) and "nodes" in subgraph_input:
                        for node in subgraph_input["nodes"]:
                            flat_evidence.append({
                                "node_id": node.get("id"),
                                "labels": [node.get("type", "Unknown")],
                                "properties": node,
                                "rel_type": "RELATED"
                            })

                # C. Generate Interactive HTML Graph
                if flat_evidence and len(flat_evidence) > 0:
                    graph_html_content = visualize_evidence_subgraph(flat_evidence, company_name=target_company, height="400px")
                    
                    if graph_html_content:
                        # ENCODE TO BASE64 to avoid quoting/escaping issues
                        b64_html = base64.b64encode(graph_html_content.encode('utf-8')).decode('utf-8')
                        
                        evidence_html += f"""
                        <div style="margin-bottom:15px; border:1px solid #ddd; border-radius:8px; overflow:hidden;">
                            <iframe 
                                src="data:text/html;base64,{b64_html}" 
                                width="100%" 
                                height="410px" 
                                style="border:none;"
                                scrolling="no">
                            </iframe>
                        </div>
                        """
                        graph_generated = True

            except Exception as e:
                # Print error to terminal so you can debug what actually broke
                print(f"Graph rendering failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. IF NO GRAPH, TRY PASSAGES
        if not graph_generated and passages:
            evidence_html += '<div style="max-height: 400px; overflow-y: auto; padding-right: 5px;">'
            for p in passages:
                snip_id = p.get("snippet_id", "?")
                content = p.get("content", "")
                evidence_html += f"""<div class="html-ev-card"><div class="html-ev-meta"><span>Passage {snip_id}</span></div><div style="font-size:0.9rem; line-height:1.4;">"{content}"</div></div>"""
            evidence_html += '</div>'

        # 3. IF NEITHER, SHOW NO CONTEXT
        elif not graph_generated and not passages:
            evidence_html += '<div style="color:#666; font-style:italic; padding:10px; background:#f0f7ff; border-radius:6px;">No direct textual evidence or graph context retrieved.</div>'

        # --- FULL CARD HTML (Flattened) ---
        full_card_html = f"""<div class="result-card">
<div class="result-header">Results</div>
<div class="result-body">
<div class="claim-wrapper">
<div class="claim-text-col">
<div class="claim-label">CLAIM UNDER REVIEW</div>
<div class="claim-content">"{res_data.get('claim', '')}"</div>
<div class="claim-meta">Company: {res_data.get('company_display')} | Year: {res_data.get('year')} | Pipeline: {res_data.get('pipeline')}</div>
</div>
<div class="verdict-box {v_class}">{v_text}</div>
</div>

<details class="custom-details" open>
<summary class="custom-summary">Justification</summary>
<div class="details-content" style="max-height: 200px; overflow-y: auto;">
    {justification_text}
</div>
</details>

<details class="custom-details" open>
<summary class="custom-summary">Context & Evidence</summary>
<div class="details-content">{evidence_html}</div>
</details>
</div>
</div>"""
        
        st.markdown(full_card_html, unsafe_allow_html=True)
        
        # --- SAVE AS JSON BUTTON ---
        json_str = json.dumps(res_data, indent=4, ensure_ascii=False)
        timestamp = res_data.get("timestamp", int(time.time()))
        
        # Create columns: Large spacer on left (80%), Button on right (20%)
        # This pushes the button to the far right.
        btn_spacer, btn_col = st.columns([0.8, 0.2])
        
        with btn_col:
            st.download_button(
                label="💾 Save Result",
                data=json_str,
                file_name=f"emerald_run_{timestamp}.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )

    elif not st.session_state.get("result_data"):
        # Optional: Show a placeholder or welcome message if no results yet
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #888;">
            <h3>Welcome to EmeraldApp</h3>
            <p>Select a company and enter a claim on the left to begin.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# RUN LOGIC
# ============================================
if st.session_state.get("show_parsing", False):
    try:
        parser = get_parser()
        for up_file in st.session_state.uploaded_files_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: tmp.write(up_file.getvalue()); tmp_path = tmp.name
            real_name = up_file.name; final_path = Path(tmp_path).parent / real_name
            if Path(tmp_path).exists(): Path(tmp_path).rename(final_path)
            
            parser.add_single_file(str(final_path))
            
            base_name = real_name.rsplit(".", 1)[0]
            parts = base_name.split("_")
            if len(parts) >= 2:
                f_year = parts[-1]
                f_comp = "_".join(parts[:-1])
                update_company_in_yaml(f_comp, f_year)
            
            st.session_state.added_files.append(real_name)
            if final_path.exists(): final_path.unlink()
            
        st.session_state.file_uploader_key += 1; st.success("Files processed!")
    except Exception as e: st.error(f"Processing failed: {e}")
    st.session_state.show_parsing = False; st.session_state.uploaded_files_data = None; st.rerun()

if st.session_state.get("show_loading", False):
    c_in, c_disp, c_yr, c_pipe, c_fs = st.session_state.claim_input, st.session_state.company_select, st.session_state.year_select, st.session_state.pipeline, st.session_state.few_shot
    c_key = COMPANY_KEY_MAP.get(c_disp, c_disp.lower())
    
    # Sanitize year: If "None" string was selected, convert to None type for backend
    if c_yr == "None": 
        c_yr = None

    try:
        if c_pipe == "EM-Baseline": res = baseline_run.run(c_in, c_fs)
        elif c_pipe == "EM-RAG": res = em_rag_run.run(c_in, company_name=c_key, year=c_yr, few_shot_flag=c_fs)
        elif c_pipe == "EM-KGRAG": res = kg_rag_run.run(c_in, company_name=c_key, year=c_yr, few_shot_flag=c_fs)
        elif c_pipe == "EM-HYBRID": res = hybrid_run.run(c_in, company_name=c_key, year=c_yr, few_shot_flag=c_fs)
        
        if res:
            pkg = {"claim": c_in, "company_display": c_disp, "company_key": c_key, "year": c_yr, "pipeline": c_pipe, "few_shot": c_fs, "result": res}
            save_run_to_history(pkg); st.session_state.result_data = pkg
    except Exception as e: st.error(f"Error: {e}")
    st.session_state.show_loading = False; st.rerun()

# ============================================
# FOOTER
# ============================================

# Load images (Assuming these files exist in your root directory)
site_logo_b64 = get_img_as_base64("site_logo.png")
athena_b64 = get_img_as_base64("ath.png")
archimedes_b64 = get_img_as_base64("archimedes.png")

footer_html = f"""
<style>
    .footer-container {{
        margin-top: 40px;
        padding: 12px 0;
        border-top: 1px solid #e0e0e0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 24px;
        width: 100%;
    }}
    .footer-main-link {{
        display: flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
        color: #435334;
        font-weight: 600;
        font-size: 0.85rem;
        transition: opacity 0.2s;
    }}
    .footer-main-link:hover {{
        opacity: 0.7;
        text-decoration: none;
    }}
    .footer-logo {{
        height: 28px; 
        object-fit: contain;
        opacity: 0.85;
        transition: opacity 0.2s;
    }}
    .footer-logo:hover {{
        opacity: 1;
    }}
    .footer-divider {{
        width: 1px;
        height: 20px;
        background: #ccc;
    }}
</style>

<div class="footer-container">
    <a href="https://sites.google.com/view/ai-against-greenwashing/" target="_blank" class="footer-main-link">
        <img src="{site_logo_b64}" class="footer-logo" alt="Logo">
        <span>AI Against Greenwashing</span>
    </a>
    <div class="footer-divider"></div>
    <a href="https://www.athenarc.gr/" target="_blank" title="Athena Research Center">
        <img src="{athena_b64}" class="footer-logo" alt="Athena RC">
    </a>
    <a href="https://archimedesai.gr/en/" target="_blank" title="Archimedes AI">
        <img src="{archimedes_b64}" class="footer-logo" alt="Archimedes AI">
    </a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)