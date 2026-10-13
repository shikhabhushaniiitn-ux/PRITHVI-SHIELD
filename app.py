
import streamlit as st

st.set_page_config(
    page_title="PRITHVI-SHIELD",
    page_icon="🌏",
    layout="wide",
)

st.title("🌏 PRITHVI-SHIELD")
st.subheader("Multimodal AI for Disaster Intelligence")

st.markdown(
    """
    A research and educational prototype for satellite-based
    flood mapping, infrastructure exposure, and community
    accessibility analysis.
    """
)

st.info(
    "Development status: Initial application scaffold. "
    "No real satellite analysis has been implemented yet."
)

st.header("Disaster Scenario")

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox(
        "Select a pilot region",
        [
            "Trishuli River Corridor, Nepal",
            "Other region (planned)",
        ],
    )

with col2:
    hazard = st.selectbox(
        "Select hazard",
        [
            "Flood",
            "Other hazards (planned)",
        ],
    )

st.divider()

st.header("Analysis Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Flood Mapping")
    st.write(
        "Planned: compare permitted pre-event and "
        "post-event satellite observations."
    )

with col2:
    st.subheader("2. Asset Exposure")
    st.write(
        "Planned: analyse mapped buildings, roads, "
        "and bridges."
    )

with col3:
    st.subheader("3. Accessibility")
    st.write(
        "Planned: analyse road connectivity to "
        "settlements and essential facilities."
    )

st.divider()

st.caption(
    f"Selected region: {region} | Hazard: {hazard}"
)

st.warning(
    "This prototype does not provide operational "
    "disaster warnings. All future estimates will "
    "require appropriate validation."
)