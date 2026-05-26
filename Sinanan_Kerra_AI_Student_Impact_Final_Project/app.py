import os
import pandas as pd
import plotly.express as px
import streamlit as st
from databricks import sql

# ============================================================
# Final Project App: AI Student Impact Explorer
# Kerra Sinanan
# ============================================================

st.set_page_config(
    page_title="AI Student Impact Explorer",
    page_icon="🎓",
    layout="wide"
)

TABLE_NAME = "ai_student_impact_clean"


# ------------------------------------------------------------
# Databricks SQL connection
# ------------------------------------------------------------

@st.cache_resource
def get_connection():
    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    access_token = os.getenv("DATABRICKS_TOKEN")

    if not server_hostname or not http_path:
        st.error("Missing Databricks SQL warehouse connection. Check app.yaml.")
        st.stop()

    if access_token:
        return sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token
        )

    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path
    )


@st.cache_data(ttl=300)
def run_query(query: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows, columns=columns)


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def get_distinct(column: str):
    df = run_query(f"""
        SELECT DISTINCT {column} AS value
        FROM {TABLE_NAME}
        WHERE {column} IS NOT NULL
        ORDER BY {column}
    """)
    return df["value"].dropna().tolist()


def build_in_filter(column: str, selected_values, all_label: str):
    if all_label in selected_values or len(selected_values) == 0:
        return None
    quoted = ", ".join(sql_quote(v) for v in selected_values)
    return f"{column} IN ({quoted})"


# ------------------------------------------------------------
# App header
# ------------------------------------------------------------

st.title("🎓 AI Student Impact Explorer")
st.write(
    "This interactive dashboard explores how Generative AI use relates to GPA change, "
    "skill retention, anxiety, dependency, and burnout risk among students."
)

try:
    major_options = get_distinct("major_category")
    year_options = get_distinct("year_of_study")
    use_case_options = get_distinct("primary_use_case")
    skill_options = get_distinct("prompt_engineering_skill")
    policy_options = get_distinct("institutional_policy")
    burnout_options = get_distinct("burnout_risk_level")
except Exception as e:
    st.error("The app could not find the cleaned table.")
    st.write("Run the final project notebook first so `ai_student_impact_clean` exists.")
    st.exception(e)
    st.stop()


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    selected_majors = st.multiselect(
        "Major category",
        options=["All Majors"] + major_options,
        default=["All Majors"]
    )

    selected_years = st.multiselect(
        "Year of study",
        options=["All Years"] + year_options,
        default=["All Years"]
    )

    selected_use_cases = st.multiselect(
        "Primary AI use case",
        options=["All Use Cases"] + use_case_options,
        default=["All Use Cases"]
    )

    selected_skills = st.multiselect(
        "Prompt engineering skill",
        options=["All Skill Levels"] + skill_options,
        default=["All Skill Levels"]
    )

    selected_policies = st.multiselect(
        "Institutional policy",
        options=["All Policies"] + policy_options,
        default=["All Policies"]
    )

    selected_burnout = st.multiselect(
        "Burnout risk level",
        options=["All Burnout Levels"] + burnout_options,
        default=["All Burnout Levels"]
    )

    genai_range = st.slider(
        "Weekly GenAI hours",
        min_value=0.0,
        max_value=40.0,
        value=(0.0, 40.0),
        step=1.0
    )


conditions = [
    f"weekly_genai_hours BETWEEN {genai_range[0]} AND {genai_range[1]}"
]

filter_specs = [
    ("major_category", selected_majors, "All Majors"),
    ("year_of_study", selected_years, "All Years"),
    ("primary_use_case", selected_use_cases, "All Use Cases"),
    ("prompt_engineering_skill", selected_skills, "All Skill Levels"),
    ("institutional_policy", selected_policies, "All Policies"),
    ("burnout_risk_level", selected_burnout, "All Burnout Levels"),
]

for column, values, all_label in filter_specs:
    condition = build_in_filter(column, values, all_label)
    if condition:
        conditions.append(condition)

where_clause = " AND ".join(conditions)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

overview_tab, ai_tab, burnout_tab, raw_tab = st.tabs(
    ["Executive Summary", "AI Usage Analysis", "Burnout & Retention", "Filtered Data"]
)


# ============================================================
# Executive Summary
# ============================================================

with overview_tab:
    st.header("Executive Summary")

    summary = run_query(f"""
        SELECT
            COUNT(*) AS students,
            ROUND(AVG(weekly_genai_hours), 2) AS avg_genai_hours,
            ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
            ROUND(AVG(post_semester_gpa), 3) AS avg_post_gpa,
            ROUND(AVG(skill_retention_score), 2) AS avg_retention,
            ROUND(AVG(anxiety_level_during_exams), 2) AS avg_anxiety,
            ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct
        FROM {TABLE_NAME}
        WHERE {where_clause}
    """)

    students = int(summary.loc[0, "students"]) if not summary.empty else 0

    if students == 0:
        st.warning("No records match the selected filters. Widen your filters in the sidebar.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", f"{students:,}")
    col2.metric("Avg GenAI Hours", f"{summary.loc[0, 'avg_genai_hours']} hrs/wk")
    col3.metric("Avg GPA Change", f"{summary.loc[0, 'avg_gpa_change']}")
    col4.metric("High Burnout %", f"{summary.loc[0, 'high_burnout_pct']}%")

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Post GPA", f"{summary.loc[0, 'avg_post_gpa']}")
    col6.metric("Avg Skill Retention", f"{summary.loc[0, 'avg_retention']}")
    col7.metric("Avg Exam Anxiety", f"{summary.loc[0, 'avg_anxiety']} / 10")

    st.divider()

    st.subheader("Main Finding")
    st.write(
        "The pattern in this dataset suggests that moderate AI use is associated with better outcomes, "
        "while very heavy AI use is connected to higher dependency, higher burnout, and lower skill retention."
    )

    st.subheader("Average GPA Change by AI Usage Band")

    band_df = run_query(f"""
        SELECT
            ai_usage_band,
            COUNT(*) AS students,
            ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
            ROUND(AVG(skill_retention_score), 2) AS avg_retention,
            ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct
        FROM {TABLE_NAME}
        WHERE {where_clause}
        GROUP BY ai_usage_band
        ORDER BY
            CASE ai_usage_band
                WHEN '0-2 hours' THEN 1
                WHEN '2-5 hours' THEN 2
                WHEN '5-10 hours' THEN 3
                WHEN '10-20 hours' THEN 4
                WHEN '20+ hours' THEN 5
            END
    """)

    fig_band = px.bar(
        band_df,
        x="ai_usage_band",
        y="avg_gpa_change",
        text="avg_gpa_change",
        hover_data=["students", "avg_retention", "high_burnout_pct"],
        title="Average GPA Change by Weekly GenAI Usage Band",
        labels={
            "ai_usage_band": "Weekly GenAI Usage Band",
            "avg_gpa_change": "Average GPA Change"
        }
    )
    fig_band.update_traces(textposition="outside")
    st.plotly_chart(fig_band, use_container_width=True)


# ============================================================
# AI Usage Analysis
# ============================================================

with ai_tab:
    st.header("AI Usage Analysis")

    st.subheader("GenAI Hours vs Skill Retention")

    scatter_df = run_query(f"""
        SELECT
            weekly_genai_hours,
            skill_retention_score,
            gpa_change,
            major_category,
            year_of_study,
            burnout_risk_level,
            prompt_engineering_skill
        FROM {TABLE_NAME}
        WHERE {where_clause}
        ORDER BY RAND()
        LIMIT 5000
    """)

    fig_scatter = px.scatter(
        scatter_df,
        x="weekly_genai_hours",
        y="skill_retention_score",
        color="burnout_risk_level",
        hover_data=["major_category", "year_of_study", "prompt_engineering_skill", "gpa_change"],
        title="Weekly GenAI Hours vs Skill Retention",
        labels={
            "weekly_genai_hours": "Weekly GenAI Hours",
            "skill_retention_score": "Skill Retention Score",
            "burnout_risk_level": "Burnout Risk"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Prompt Engineering Skill Comparison")

    skill_df = run_query(f"""
        SELECT
            prompt_engineering_skill,
            COUNT(*) AS students,
            ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
            ROUND(AVG(skill_retention_score), 2) AS avg_retention,
            ROUND(AVG(tool_diversity), 2) AS avg_tool_diversity,
            ROUND(100.0 * AVG(strong_retention_flag), 2) AS strong_retention_pct
        FROM {TABLE_NAME}
        WHERE {where_clause}
        GROUP BY prompt_engineering_skill
        ORDER BY avg_gpa_change DESC
    """)

    fig_skill = px.bar(
        skill_df,
        x="prompt_engineering_skill",
        y="avg_retention",
        text="avg_retention",
        hover_data=["students", "avg_gpa_change", "avg_tool_diversity", "strong_retention_pct"],
        title="Skill Retention by Prompt Engineering Skill",
        labels={
            "prompt_engineering_skill": "Prompt Engineering Skill",
            "avg_retention": "Average Skill Retention"
        }
    )
    fig_skill.update_traces(textposition="outside")
    st.plotly_chart(fig_skill, use_container_width=True)

    st.dataframe(skill_df, use_container_width=True)


# ============================================================
# Burnout and Retention
# ============================================================

with burnout_tab:
    st.header("Burnout & Retention")

    st.subheader("Burnout Risk Distribution by AI Usage Band")

    burnout_df = run_query(f"""
        SELECT
            ai_usage_band,
            burnout_risk_level,
            COUNT(*) AS students
        FROM {TABLE_NAME}
        WHERE {where_clause}
        GROUP BY ai_usage_band, burnout_risk_level
        ORDER BY
            CASE ai_usage_band
                WHEN '0-2 hours' THEN 1
                WHEN '2-5 hours' THEN 2
                WHEN '5-10 hours' THEN 3
                WHEN '10-20 hours' THEN 4
                WHEN '20+ hours' THEN 5
            END,
            burnout_risk_level
    """)

    fig_burnout = px.bar(
        burnout_df,
        x="ai_usage_band",
        y="students",
        color="burnout_risk_level",
        barmode="stack",
        title="Burnout Risk Distribution by AI Usage Band",
        labels={
            "ai_usage_band": "Weekly GenAI Usage Band",
            "students": "Students",
            "burnout_risk_level": "Burnout Risk"
        }
    )
    st.plotly_chart(fig_burnout, use_container_width=True)

    st.subheader("Institutional Policy Outcomes")

    policy_df = run_query(f"""
        SELECT
            institutional_policy,
            COUNT(*) AS students,
            ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
            ROUND(AVG(skill_retention_score), 2) AS avg_retention,
            ROUND(AVG(anxiety_level_during_exams), 2) AS avg_anxiety,
            ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct,
            ROUND(100.0 * AVG(high_dependency_flag), 2) AS high_dependency_pct
        FROM {TABLE_NAME}
        WHERE {where_clause}
        GROUP BY institutional_policy
        ORDER BY high_burnout_pct DESC
    """)

    fig_policy = px.bar(
        policy_df,
        x="institutional_policy",
        y="high_burnout_pct",
        text="high_burnout_pct",
        hover_data=["students", "avg_gpa_change", "avg_retention", "avg_anxiety", "high_dependency_pct"],
        title="High Burnout Percentage by Institutional Policy",
        labels={
            "institutional_policy": "Institutional Policy",
            "high_burnout_pct": "High Burnout %"
        }
    )
    fig_policy.update_traces(textposition="outside")
    st.plotly_chart(fig_policy, use_container_width=True)

    st.dataframe(policy_df, use_container_width=True)


# ============================================================
# Filtered Data
# ============================================================

with raw_tab:
    st.header("Filtered Data")

    detail_df = run_query(f"""
        SELECT
            student_id,
            major_category,
            year_of_study,
            pre_semester_gpa,
            post_semester_gpa,
            gpa_change,
            weekly_genai_hours,
            ai_usage_band,
            primary_use_case,
            prompt_engineering_skill,
            tool_diversity,
            traditional_study_hours,
            perceived_ai_dependency,
            anxiety_level_during_exams,
            skill_retention_score,
            burnout_risk_level,
            institutional_policy
        FROM {TABLE_NAME}
        WHERE {where_clause}
        ORDER BY gpa_change DESC
        LIMIT 1000
    """)

    st.dataframe(detail_df, use_container_width=True)

    csv = detail_df.to_csv(index=False)
    st.download_button(
        label="Download filtered student data as CSV",
        data=csv,
        file_name="filtered_ai_student_impact.csv",
        mime="text/csv"
    )

st.caption(
    "Final Project: AI Student Impact Explorer. Built with Databricks Apps, Streamlit, SQL, pandas, and Plotly."
)
