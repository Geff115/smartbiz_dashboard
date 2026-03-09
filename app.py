import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import anthropic
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartBiz Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (clean, professional look)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .alert-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def format_naira(value: float) -> str:
    """Format a number as Nigerian Naira."""
    return f"₦{value:,.0f}"


def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names and parse dates.
    Expected columns (case-insensitive):
        date, product, category, quantity, unit_price, revenue, stock_remaining
    """
    # Lowercase all column names and strip whitespace
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["day_of_week"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.to_period("M").astype(str)

    # Calculate revenue if not present but quantity & unit_price are
    if "revenue" not in df.columns:
        if "quantity" in df.columns and "unit_price" in df.columns:
            df["revenue"] = df["quantity"] * df["unit_price"]

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of key business metrics."""
    kpis = {}
    if "revenue" in df.columns:
        kpis["total_revenue"] = df["revenue"].sum()
        kpis["avg_order_value"] = df["revenue"].mean()
        kpis["total_transactions"] = len(df)
    if "quantity" in df.columns:
        kpis["total_units_sold"] = df["quantity"].sum()
    if "product" in df.columns:
        kpis["unique_products"] = df["product"].nunique()
    return kpis


def get_low_stock_alerts(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """Return products whose stock is at or below the threshold."""
    if "stock_remaining" not in df.columns or "product" not in df.columns:
        return pd.DataFrame()
    stock_df = (
        df.groupby("product")["stock_remaining"]
        .min()
        .reset_index()
        .rename(columns={"stock_remaining": "stock_left"})
    )
    return stock_df[stock_df["stock_left"] <= threshold].sort_values("stock_left")


def generate_ai_insights(df: pd.DataFrame, api_key: str) -> str:
    """
    Send a data summary to Claude and get plain-English business insights.
    """
    # Build a concise summary to send to the API (keeps token cost low)
    summary_parts = []

    if "revenue" in df.columns:
        summary_parts.append(f"Total revenue: ₦{df['revenue'].sum():,.0f}")
        summary_parts.append(f"Average transaction value: ₦{df['revenue'].mean():,.0f}")

    if "product" in df.columns and "revenue" in df.columns:
        top5 = (
            df.groupby("product")["revenue"]
            .sum()
            .nlargest(5)
            .reset_index()
            .to_string(index=False)
        )
        summary_parts.append(f"\nTop 5 products by revenue:\n{top5}")

    if "day_of_week" in df.columns and "revenue" in df.columns:
        by_day = (
            df.groupby("day_of_week")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .to_string(index=False)
        )
        summary_parts.append(f"\nRevenue by day of week:\n{by_day}")

    if "category" in df.columns and "revenue" in df.columns:
        by_cat = (
            df.groupby("category")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .to_string(index=False)
        )
        summary_parts.append(f"\nRevenue by category:\n{by_cat}")

    low_stock = get_low_stock_alerts(df)
    if not low_stock.empty:
        summary_parts.append(f"\nLow stock products:\n{low_stock.to_string(index=False)}")

    full_summary = "\n".join(summary_parts)

    prompt = f"""You are a friendly and experienced business analyst for small businesses in Lagos, Nigeria.
Analyze the sales data below and give exactly 4 specific, actionable insights.

Rules:
- Write in simple, warm English — no jargon
- Be specific (mention product names, days, numbers from the data)
- Each insight should start with a relevant emoji
- Keep each insight to 2 sentences max
- End with one motivational sentence for the business owner

Data:
{full_summary}
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ Could not generate insights: {str(e)}"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("SmartBiz Dashboard")
    st.caption("Powered by Python + Claude AI")
    st.divider()

    st.subheader("⚙️ Settings")
    api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )

    st.divider()
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Drop your sales file here",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file with columns: date, product, category, quantity, unit_price, revenue, stock_remaining",
    )

    st.divider()
    low_stock_threshold = st.slider(
        "Low Stock Alert Threshold",
        min_value=1, max_value=50, value=10,
        help="Products with stock at or below this number will be flagged"
    )

    st.divider()
    st.caption("Built by a SmartBiz Developer 🚀")


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.title("📊 SmartBiz Dashboard")
st.caption("Upload your sales data to unlock insights about your business.")

if uploaded_file is None:
    # ── Welcome / landing state ──
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")

    st.markdown("### 📋 Expected File Format")
    sample = pd.DataFrame({
        "date":            ["2024-01-01", "2024-01-01", "2024-01-02"],
        "product":         ["Indomie Carton", "Milo 400g", "Dangote Sugar"],
        "category":        ["Food", "Beverages", "Food"],
        "quantity":        [10, 5, 8],
        "unit_price":      [4500, 3200, 2800],
        "revenue":         [45000, 16000, 22400],
        "stock_remaining": [40, 12, 7],
    })
    st.dataframe(sample, use_container_width=True)
    st.caption("Your file doesn't have to be perfect — the dashboard will adapt to what's available.")
    st.stop()


# ── Load & clean data ──
df = load_data(uploaded_file)
df = clean_data(df)
kpis = compute_kpis(df)

st.success(f"✅ Loaded **{len(df):,} records** from `{uploaded_file.name}`")

# ─────────────────────────────────────────────
#  KPI METRICS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

cols = st.columns(5)
metric_items = [
    ("💰 Total Revenue",       format_naira(kpis.get("total_revenue", 0))),
    ("🧾 Avg Transaction",     format_naira(kpis.get("avg_order_value", 0))),
    ("📦 Total Units Sold",    f"{int(kpis.get('total_units_sold', 0)):,}"),
    ("🔢 Transactions",        f"{kpis.get('total_transactions', 0):,}"),
    ("🛍️ Unique Products",     str(kpis.get("unique_products", 0))),
]
for col, (label, value) in zip(cols, metric_items):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHARTS  (2-column layout)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Sales Visuals</div>', unsafe_allow_html=True)

left, right = st.columns(2)

# ── Revenue over time ──
with left:
    if "date" in df.columns and "revenue" in df.columns:
        daily = df.groupby("date")["revenue"].sum().reset_index()
        fig = px.area(
            daily, x="date", y="revenue",
            title="💰 Revenue Over Time",
            labels={"revenue": "Revenue (₦)", "date": "Date"},
            color_discrete_sequence=["#667eea"],
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            title_font_size=14, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Top 10 products ──
with right:
    if "product" in df.columns and "revenue" in df.columns:
        top10 = (
            df.groupby("product")["revenue"]
            .sum()
            .nlargest(10)
            .sort_values()
            .reset_index()
        )
        fig = px.bar(
            top10, x="revenue", y="product",
            orientation="h",
            title="🏆 Top 10 Products by Revenue",
            labels={"revenue": "Revenue (₦)", "product": "Product"},
            color="revenue",
            color_continuous_scale="Purples",
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            title_font_size=14, margin=dict(t=40, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


left2, right2 = st.columns(2)

# ── Sales by day of week ──
with left2:
    if "day_of_week" in df.columns and "revenue" in df.columns:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = (
            df.groupby("day_of_week")["revenue"]
            .sum()
            .reindex(day_order)
            .reset_index()
        )
        fig = px.bar(
            by_day, x="day_of_week", y="revenue",
            title="📅 Revenue by Day of Week",
            labels={"revenue": "Revenue (₦)", "day_of_week": "Day"},
            color="revenue",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            title_font_size=14, margin=dict(t=40, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Revenue by category (pie) ──
with right2:
    if "category" in df.columns and "revenue" in df.columns:
        by_cat = df.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(
            by_cat, names="category", values="revenue",
            title="🗂️ Revenue by Category",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            hole=0.4,
        )
        fig.update_layout(title_font_size=14, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  INVENTORY ALERTS
# ─────────────────────────────────────────────
low_stock = get_low_stock_alerts(df, threshold=low_stock_threshold)
if not low_stock.empty:
    st.markdown('<div class="section-header">⚠️ Low Stock Alerts</div>', unsafe_allow_html=True)
    for _, row in low_stock.iterrows():
        urgency = "🔴" if row["stock_left"] <= 3 else "🟡"
        st.markdown(
            f'<div class="alert-box">{urgency} <strong>{row["product"]}</strong> — '
            f'only <strong>{int(row["stock_left"])} units</strong> remaining. Restock soon!</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
#  AI INSIGHTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 AI-Generated Insights</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("🔑 Add your Claude API key in the sidebar to unlock AI-powered insights.")
else:
    if st.button("✨ Generate Insights", type="primary", use_container_width=True):
        with st.spinner("Analyzing your data with Claude AI..."):
            insights = generate_ai_insights(df, api_key)
        st.markdown(
            f'<div class="insight-box"><h4>📋 Business Insights</h4>{insights.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
#  RAW DATA VIEWER
# ─────────────────────────────────────────────
with st.expander("🔍 View Raw Data"):
    st.dataframe(df, use_container_width=True)
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cleaned Data as CSV",
        data=csv_out,
        file_name="smartbiz_cleaned_data.csv",
        mime="text/csv",
    )
