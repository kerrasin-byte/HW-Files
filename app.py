import os
import pandas as pd
import plotly.express as px
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

# ============================================================
# Week 7 Assignment: Flight Delay Explorer + Diamond Explorer
# Kerra Sinanan
# ============================================================

st.set_page_config(
    page_title="Flight Delay Explorer",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------------------
# Databricks SQL execution using SDK Statement API
# ------------------------------------------------------------

@st.cache_resource
def get_workspace_client():
    """Get authenticated Databricks workspace client."""
    return WorkspaceClient()


@st.cache_data(ttl=300)
def run_query(query: str) -> pd.DataFrame:
    """
    Run a SQL query using Databricks SQL Statement API.
    """
    try:
        w = get_workspace_client()
        warehouse_id = os.getenv("WAREHOUSE_ID", "264d126b27bbf34e")
        
        # Execute SQL statement
        statement = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query
        )
        
        # Check if execution succeeded
        if statement.status.state != StatementState.SUCCEEDED:
            error_msg = statement.status.error.message if statement.status.error else "Unknown error"
            st.error(f"Query failed: {error_msg}")
            return pd.DataFrame()
        
        # Convert result to pandas DataFrame
        if statement.result and statement.result.data_array:
            columns = [col.name for col in statement.manifest.schema.columns]
            rows = statement.result.data_array
            return pd.DataFrame(rows, columns=columns)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        return pd.DataFrame()


def sql_quote(value: str) -> str:
    """Safely quote a string value for SQL filters."""
    return "'" + str(value).replace("'", "''") + "'"


def empty_message(message: str):
    st.warning(message)
    st.stop()


# ------------------------------------------------------------
# App title
# ------------------------------------------------------------

st.title("✈️ Flight Delay Explorer")
st.write(
    "An interactive Databricks Streamlit dashboard for exploring NYC flight delays "
    "and diamond pricing patterns."
)

# These table names match the class examples.
FLIGHTS_TABLE = "workspace.default.flights"
AIRLINES_TABLE = "workspace.default.airlines"
DIAMONDS_TABLE = "workspace.default.diamonds"

# ------------------------------------------------------------
# Load reference data
# ------------------------------------------------------------

try:
    airlines = run_query(f"""
        SELECT carrier, name
        FROM {AIRLINES_TABLE}
        ORDER BY name
    """)

    cuts = run_query(f"""
        SELECT DISTINCT cut
        FROM {DIAMONDS_TABLE}
        WHERE cut IS NOT NULL
        ORDER BY cut
    """)

except Exception as e:
    st.error("The app could not load the required Databricks tables.")
    st.write("Check that these tables exist: `flights`, `airlines`, and `diamonds`.")
    st.exception(e)
    st.stop()

if airlines.empty:
    empty_message("No airline data was found.")

airline_options = dict(zip(airlines["name"], airlines["carrier"]))
cut_options = cuts["cut"].tolist() if not cuts.empty else []


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------

with st.sidebar:
    st.header("Flight Filters")

    selected_airline = st.selectbox(
        "Airline",
        options=["All Airlines"] + list(airline_options.keys())
    )

    month_range = st.slider(
        "Month range",
        min_value=1,
        max_value=12,
        value=(1, 12)
    )

    delay_type = st.radio(
        "Delay metric",
        options=["Arrival Delay", "Departure Delay"],
        index=0
    )

    min_delay = st.slider(
        "Minimum delay to include",
        min_value=-60,
        max_value=300,
        value=-60,
        step=5
    )

    st.divider()

    st.header("Diamond Filters")

    selected_cut = st.selectbox(
        "Diamond cut",
        options=["All Cuts"] + cut_options
    )

    max_points = st.slider(
        "Scatter plot sample size",
        min_value=500,
        max_value=10000,
        value=3000,
        step=500
    )


delay_col = "arr_delay" if delay_type == "Arrival Delay" else "dep_delay"
delay_label = "Arrival Delay" if delay_col == "arr_delay" else "Departure Delay"

flight_conditions = [
    f"f.{delay_col} IS NOT NULL",
    f"f.month BETWEEN {month_range[0]} AND {month_range[1]}",
    f"f.{delay_col} >= {min_delay}"
]

if selected_airline != "All Airlines":
    carrier_code = airline_options[selected_airline]
    flight_conditions.append(f"f.carrier = {sql_quote(carrier_code)}")

flight_where = " AND ".join(flight_conditions)

diamond_conditions = ["price IS NOT NULL", "carat IS NOT NULL", "cut IS NOT NULL"]

if selected_cut != "All Cuts":
    diamond_conditions.append(f"cut = {sql_quote(selected_cut)}")

diamond_where = " AND ".join(diamond_conditions)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

flights_tab, diamonds_tab = st.tabs(["Flights Dashboard", "Diamond Price Explorer"])


# ============================================================
# TAB 1: FLIGHT DELAY EXPLORER
# ============================================================

with flights_tab:
    st.header("Flight Delay Dashboard")
    st.write(
        "Use the sidebar to filter by airline, month range, delay type, and minimum delay."
    )

    # Summary metrics
    summary = run_query(f"""
        SELECT
            COUNT(*) AS total_flights,
            ROUND(AVG(f.{delay_col}), 1) AS avg_delay,
            ROUND(MIN(f.{delay_col}), 1) AS best_delay,
            ROUND(MAX(f.{delay_col}), 1) AS worst_delay,
            ROUND(
                100.0 * SUM(CASE WHEN f.{delay_col} <= 0 THEN 1 ELSE 0 END) / COUNT(*),
                1
            ) AS ontime_pct
        FROM {FLIGHTS_TABLE} f
        WHERE {flight_where}
    """)

    total_flights = int(summary.loc[0, "total_flights"]) if not summary.empty else 0

    if total_flights == 0:
        st.warning("No flights match the selected filters. Try widening the month or delay range.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flights", f"{total_flights:,}")
        col2.metric(f"Average {delay_label}", f"{summary.loc[0, 'avg_delay']} min")
        col3.metric("On-Time %", f"{summary.loc[0, 'ontime_pct']}%")
        col4.metric("Worst Delay", f"{summary.loc[0, 'worst_delay']} min")

        st.divider()

        # Chart 1: Delay by month
        st.subheader("Average Delay by Month")

        monthly = run_query(f"""
            SELECT
                f.month,
                ROUND(AVG(f.{delay_col}), 2) AS avg_delay,
                COUNT(*) AS flights
            FROM {FLIGHTS_TABLE} f
            WHERE {flight_where}
            GROUP BY f.month
            ORDER BY f.month
        """)

        fig_month = px.bar(
            monthly,
            x="month",
            y="avg_delay",
            text="avg_delay",
            title=f"Average {delay_label} by Month",
            labels={
                "month": "Month",
                "avg_delay": "Average Delay (minutes)",
                "flights": "Flights"
            },
            hover_data=["flights"]
        )
        fig_month.update_traces(textposition="outside")
        fig_month.update_layout(
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis_title="Average Delay (minutes)",
            showlegend=False
        )
        st.plotly_chart(fig_month, use_container_width=True)

        # Chart 2: Delay distribution
        st.subheader("Delay Distribution")

        delays = run_query(f"""
            SELECT f.{delay_col} AS delay
            FROM {FLIGHTS_TABLE} f
            WHERE {flight_where}
              AND f.{delay_col} BETWEEN -60 AND 240
        """)

        fig_dist = px.histogram(
            delays,
            x="delay",
            nbins=60,
            title=f"{delay_label} Distribution",
            labels={"delay": f"{delay_label} (minutes)"}
        )
        fig_dist.update_layout(
            xaxis_title=f"{delay_label} (minutes)",
            yaxis_title="Number of Flights"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Extra feature 1: origin airport share pie chart
        st.subheader("Extra Feature: Share of Flights by Origin Airport")

        origin_share = run_query(f"""
            SELECT
                f.origin,
                COUNT(*) AS flights
            FROM {FLIGHTS_TABLE} f
            WHERE {flight_where}
            GROUP BY f.origin
            ORDER BY flights DESC
        """)

        fig_origin = px.pie(
            origin_share,
            names="origin",
            values="flights",
            title="Flight Share by Origin Airport"
        )
        st.plotly_chart(fig_origin, use_container_width=True)

        # Extra feature 2: worst delays table
        st.subheader("Extra Feature: Top 10 Worst Delayed Flights")

        worst_delays = run_query(f"""
            SELECT
                f.year,
                f.month,
                f.day,
                f.carrier,
                a.name AS airline,
                f.flight,
                f.origin,
                f.dest,
                f.{delay_col} AS selected_delay,
                f.dep_delay,
                f.arr_delay,
                f.air_time
            FROM {FLIGHTS_TABLE} f
            LEFT JOIN {AIRLINES_TABLE} a
                ON f.carrier = a.carrier
            WHERE {flight_where}
            ORDER BY f.{delay_col} DESC
            LIMIT 10
        """)

        st.dataframe(worst_delays, use_container_width=True)

        st.divider()

        # Airline comparison table + download
        st.subheader("Airline Comparison Table")

        carrier_stats = run_query(f"""
            SELECT
                a.name AS airline,
                f.carrier,
                COUNT(*) AS flights,
                ROUND(AVG(f.{delay_col}), 1) AS avg_delay,
                ROUND(MIN(f.{delay_col}), 1) AS best_delay,
                ROUND(MAX(f.{delay_col}), 1) AS worst_delay,
                ROUND(
                    100.0 * SUM(CASE WHEN f.{delay_col} <= 0 THEN 1 ELSE 0 END) / COUNT(*),
                    1
                ) AS ontime_pct
            FROM {FLIGHTS_TABLE} f
            LEFT JOIN {AIRLINES_TABLE} a
                ON f.carrier = a.carrier
            WHERE f.{delay_col} IS NOT NULL
              AND f.month BETWEEN {month_range[0]} AND {month_range[1]}
              AND f.{delay_col} >= {min_delay}
            GROUP BY a.name, f.carrier
            ORDER BY avg_delay
        """)

        st.dataframe(carrier_stats, use_container_width=True)

        csv = carrier_stats.to_csv(index=False)
        st.download_button(
            label="Download airline comparison as CSV",
            data=csv,
            file_name="airline_delay_stats.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 2: DIAMOND PRICE EXPLORER
# ============================================================

with diamonds_tab:
    st.header("Diamond Price Explorer")
    st.write(
        "This tab explores the relationship between diamond carat, cut, and price."
    )

    diamond_summary = run_query(f"""
        SELECT
            COUNT(*) AS total_diamonds,
            ROUND(AVG(price), 2) AS avg_price,
            ROUND(AVG(carat), 2) AS avg_carat,
            ROUND(MIN(price), 2) AS min_price,
            ROUND(MAX(price), 2) AS max_price
        FROM {DIAMONDS_TABLE}
        WHERE {diamond_where}
    """)

    if int(diamond_summary.loc[0, "total_diamonds"]) == 0:
        st.warning("No diamonds match the selected cut filter.")
    else:
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Total Diamonds", f"{int(diamond_summary.loc[0, 'total_diamonds']):,}")
        dcol2.metric("Average Price", f"${diamond_summary.loc[0, 'avg_price']:,.2f}")
        dcol3.metric("Average Carat", f"{diamond_summary.loc[0, 'avg_carat']}")
        dcol4.metric("Max Price", f"${diamond_summary.loc[0, 'max_price']:,.2f}")

        st.divider()

        st.subheader("Carat vs Price Scatter Plot")

        diamond_points = run_query(f"""
            SELECT
                carat,
                price,
                cut,
                color,
                clarity
            FROM {DIAMONDS_TABLE}
            WHERE {diamond_where}
            ORDER BY RAND()
            LIMIT {max_points}
        """)

        fig_scatter = px.scatter(
            diamond_points,
            x="carat",
            y="price",
            color="cut",
            hover_data=["color", "clarity"],
            title="Diamond Price by Carat and Cut",
            labels={
                "carat": "Carat",
                "price": "Price ($)",
                "cut": "Cut"
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Average Price by Cut")

        price_by_cut = run_query(f"""
            SELECT
                cut,
                COUNT(*) AS diamonds,
                ROUND(AVG(price), 2) AS avg_price,
                ROUND(AVG(carat), 2) AS avg_carat
            FROM {DIAMONDS_TABLE}
            WHERE price IS NOT NULL
              AND carat IS NOT NULL
              AND cut IS NOT NULL
            GROUP BY cut
            ORDER BY avg_price DESC
        """)

        fig_cut = px.bar(
            price_by_cut,
            x="cut",
            y="avg_price",
            text="avg_price",
            title="Average Diamond Price by Cut",
            labels={
                "cut": "Cut",
                "avg_price": "Average Price ($)"
            },
            hover_data=["diamonds", "avg_carat"]
        )
        fig_cut.update_traces(textposition="outside")
        st.plotly_chart(fig_cut, use_container_width=True)

        st.dataframe(price_by_cut, use_container_width=True)

        diamond_csv = diamond_points.to_csv(index=False)
        st.download_button(
            label="Download filtered diamond sample as CSV",
            data=diamond_csv,
            file_name="diamond_price_sample.csv",
            mime="text/csv"
        )
