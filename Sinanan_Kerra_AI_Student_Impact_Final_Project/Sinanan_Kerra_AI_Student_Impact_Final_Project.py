# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Project: Generative AI Use and Student Outcomes
# MAGIC
# MAGIC **Student:** Kerra Sinanan  
# MAGIC **Course:** Business Analytics / Foundations of Empirical Research  
# MAGIC **Dataset:** AI Student Impact Dataset  
# MAGIC
# MAGIC ## Project Question
# MAGIC
# MAGIC **How does Generative AI use relate to student academic performance, skill retention, anxiety, and burnout risk?**
# MAGIC
# MAGIC This project analyzes whether different levels and types of AI usage are associated with stronger or weaker student outcomes. The main outcome variables are:
# MAGIC
# MAGIC - GPA change from pre-semester to post-semester
# MAGIC - Skill retention score
# MAGIC - Anxiety level during exams
# MAGIC - Burnout risk level
# MAGIC
# MAGIC This notebook covers the required final-project components:
# MAGIC
# MAGIC - Data loading
# MAGIC - Data exploration
# MAGIC - Data cleaning
# MAGIC - SQL analysis
# MAGIC - Pandas analysis
# MAGIC - Visualizations
# MAGIC - Findings to support a Databricks Streamlit app

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the Dataset
# MAGIC
# MAGIC Upload `ai_student_impact_dataset.csv` into Databricks first.
# MAGIC
# MAGIC Recommended upload path:
# MAGIC
# MAGIC `/FileStore/tables/ai_student_impact_dataset.csv`
# MAGIC
# MAGIC If your file is uploaded somewhere else, update the `csv_path` below.

# COMMAND ----------

from pyspark.sql.functions import (
    col, count, when, round as spark_round, avg, min as spark_min,
    max as spark_max, trim, regexp_replace, lit
)
from pyspark.sql.types import DoubleType, IntegerType, BooleanType
import re

csv_path = "/FileStore/tables/ai_student_impact_dataset.csv"

raw_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(csv_path)
)

display(raw_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initial Data Exploration
# MAGIC
# MAGIC First, inspect the shape, schema, and basic summary statistics.

# COMMAND ----------

print(f"Rows: {raw_df.count():,}")
print(f"Columns: {len(raw_df.columns):,}")
raw_df.printSchema()

# COMMAND ----------

display(raw_df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Clean Column Names
# MAGIC
# MAGIC The original dataset uses mixed capitalization. For Databricks SQL and app development, snake_case names are easier to use.

# COMMAND ----------

def clean_column_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return cleaned

clean_columns = [clean_column_name(c) for c in raw_df.columns]
df = raw_df.toDF(*clean_columns)

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Missing Value Audit
# MAGIC
# MAGIC The project requires data cleaning. Before filling or dropping anything, we check the missing-value count for every column.

# COMMAND ----------

missing_audit = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

display(missing_audit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Duplicate Check and Removal
# MAGIC
# MAGIC Duplicates can overcount students and distort averages. The student ID should be unique, so duplicates are removed based on `student_id`.

# COMMAND ----------

total_rows = df.count()
unique_students = df.select("student_id").distinct().count()
duplicate_student_ids = total_rows - unique_students

print(f"Total rows before deduplication: {total_rows:,}")
print(f"Unique student IDs: {unique_students:,}")
print(f"Duplicate student IDs found: {duplicate_student_ids:,}")

df = df.dropDuplicates(["student_id"])

print(f"Rows after deduplication: {df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Type Validation and Casting
# MAGIC
# MAGIC Numeric columns are explicitly cast so that SQL aggregations, correlations, and visualizations work correctly.

# COMMAND ----------

numeric_double_cols = [
    "pre_semester_gpa",
    "weekly_genai_hours",
    "traditional_study_hours",
    "post_semester_gpa",
    "skill_retention_score"
]

numeric_int_cols = [
    "student_id",
    "tool_diversity",
    "perceived_ai_dependency",
    "anxiety_level_during_exams"
]

for c in numeric_double_cols:
    df = df.withColumn(c, col(c).cast(DoubleType()))

for c in numeric_int_cols:
    df = df.withColumn(c, col(c).cast(IntegerType()))

df = df.withColumn("paid_subscription", col("paid_subscription").cast(BooleanType()))

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. String Cleaning
# MAGIC
# MAGIC Categorical fields are trimmed and standardized. This prevents the same category from being counted separately because of extra spaces.

# COMMAND ----------

categorical_cols = [
    "major_category",
    "year_of_study",
    "primary_use_case",
    "prompt_engineering_skill",
    "institutional_policy",
    "burnout_risk_level"
]

for c in categorical_cols:
    df = df.withColumn(c, trim(col(c)))
    df = df.withColumn(c, regexp_replace(col(c), r"\s+", "_"))

display(df.select(categorical_cols).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Fill Missing Values
# MAGIC
# MAGIC The original file has no missing values, but this step is included to show a complete cleaning pipeline. If new missing values appear in a future version of the data, numeric columns will be filled with medians and categorical columns will be filled with `"Unknown"`.

# COMMAND ----------

median_fill_values = {}

for c in numeric_double_cols + numeric_int_cols:
    median_value = df.approxQuantile(c, [0.5], 0.01)[0]
    median_fill_values[c] = median_value

df = df.fillna(median_fill_values)

categorical_fill_values = {c: "Unknown" for c in categorical_cols}
df = df.fillna(categorical_fill_values)

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Feature Engineering
# MAGIC
# MAGIC New columns are created to make the analysis and app more meaningful:
# MAGIC
# MAGIC - `gpa_change`: post-semester GPA minus pre-semester GPA
# MAGIC - `ai_usage_band`: groups students by weekly GenAI hours
# MAGIC - `high_dependency_flag`: identifies students with high perceived AI dependency
# MAGIC - `high_anxiety_flag`: identifies students with high exam anxiety
# MAGIC - `high_burnout_flag`: identifies students marked as high burnout risk
# MAGIC - `strong_retention_flag`: identifies students with skill retention score of 80 or higher

# COMMAND ----------

df = (
    df
    .withColumn("gpa_change", spark_round(col("post_semester_gpa") - col("pre_semester_gpa"), 3))
    .withColumn(
        "ai_usage_band",
        when(col("weekly_genai_hours") <= 2, "0-2 hours")
        .when(col("weekly_genai_hours") <= 5, "2-5 hours")
        .when(col("weekly_genai_hours") <= 10, "5-10 hours")
        .when(col("weekly_genai_hours") <= 20, "10-20 hours")
        .otherwise("20+ hours")
    )
    .withColumn("high_dependency_flag", when(col("perceived_ai_dependency") >= 7, 1).otherwise(0))
    .withColumn("high_anxiety_flag", when(col("anxiety_level_during_exams") >= 7, 1).otherwise(0))
    .withColumn("high_burnout_flag", when(col("burnout_risk_level") == "High", 1).otherwise(0))
    .withColumn("strong_retention_flag", when(col("skill_retention_score") >= 80, 1).otherwise(0))
)

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Outlier Detection
# MAGIC
# MAGIC This step flags unusually high GenAI usage using the IQR method.

# COMMAND ----------

q1, q3 = df.approxQuantile("weekly_genai_hours", [0.25, 0.75], 0.01)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Q1: {q1}")
print(f"Q3: {q3}")
print(f"IQR: {iqr}")
print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")

df = df.withColumn(
    "genai_hours_outlier_flag",
    when(
        (col("weekly_genai_hours") < lower_bound) |
        (col("weekly_genai_hours") > upper_bound),
        1
    ).otherwise(0)
)

display(
    df.select(
        "student_id",
        "weekly_genai_hours",
        "genai_hours_outlier_flag",
        "ai_usage_band"
    )
    .orderBy(col("weekly_genai_hours").desc())
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save Cleaned Table
# MAGIC
# MAGIC This table is used by the SQL analysis and the Streamlit app.

# COMMAND ----------

df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("ai_student_impact_clean")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS cleaned_rows
# MAGIC FROM ai_student_impact_clean;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validation Checks
# MAGIC
# MAGIC These checks confirm the cleaned dataset is ready for analysis.

# COMMAND ----------

row_count = df.count()
duplicate_ids = row_count - df.select("student_id").distinct().count()
null_student_ids = df.filter(col("student_id").isNull()).count()
bad_gpa = df.filter((col("pre_semester_gpa") < 0) | (col("pre_semester_gpa") > 4) | (col("post_semester_gpa") < 0) | (col("post_semester_gpa") > 4)).count()
bad_retention = df.filter((col("skill_retention_score") < 0) | (col("skill_retention_score") > 100)).count()
bad_ai_hours = df.filter(col("weekly_genai_hours") < 0).count()

validation_rows = [
    ("Row count >= 1,000", "PASS" if row_count >= 1000 else "FAIL", f"{row_count:,} rows"),
    ("No duplicate student IDs", "PASS" if duplicate_ids == 0 else "FAIL", f"{duplicate_ids:,} duplicate IDs"),
    ("No NULL student IDs", "PASS" if null_student_ids == 0 else "FAIL", f"{null_student_ids:,} null IDs"),
    ("GPA values between 0 and 4", "PASS" if bad_gpa == 0 else "FAIL", f"{bad_gpa:,} invalid GPA rows"),
    ("Skill retention between 0 and 100", "PASS" if bad_retention == 0 else "FAIL", f"{bad_retention:,} invalid retention rows"),
    ("Weekly GenAI hours non-negative", "PASS" if bad_ai_hours == 0 else "FAIL", f"{bad_ai_hours:,} invalid hour rows")
]

validation_df = spark.createDataFrame(validation_rows, ["check_name", "result", "detail"])
display(validation_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # SQL Analysis
# MAGIC
# MAGIC The following SQL queries answer the project question from multiple angles.

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Query 1: AI Usage Bands and Student Outcomes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     ai_usage_band,
# MAGIC     COUNT(*) AS students,
# MAGIC     ROUND(AVG(weekly_genai_hours), 2) AS avg_weekly_genai_hours,
# MAGIC     ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
# MAGIC     ROUND(AVG(post_semester_gpa), 3) AS avg_post_gpa,
# MAGIC     ROUND(AVG(skill_retention_score), 2) AS avg_skill_retention,
# MAGIC     ROUND(AVG(anxiety_level_during_exams), 2) AS avg_exam_anxiety,
# MAGIC     ROUND(AVG(perceived_ai_dependency), 2) AS avg_ai_dependency,
# MAGIC     ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct
# MAGIC FROM ai_student_impact_clean
# MAGIC GROUP BY ai_usage_band
# MAGIC ORDER BY
# MAGIC     CASE ai_usage_band
# MAGIC         WHEN '0-2 hours' THEN 1
# MAGIC         WHEN '2-5 hours' THEN 2
# MAGIC         WHEN '5-10 hours' THEN 3
# MAGIC         WHEN '10-20 hours' THEN 4
# MAGIC         WHEN '20+ hours' THEN 5
# MAGIC     END;

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Query 2: Prompt Engineering Skill and Outcomes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     prompt_engineering_skill,
# MAGIC     COUNT(*) AS students,
# MAGIC     ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
# MAGIC     ROUND(AVG(skill_retention_score), 2) AS avg_skill_retention,
# MAGIC     ROUND(AVG(tool_diversity), 2) AS avg_tool_diversity,
# MAGIC     ROUND(AVG(anxiety_level_during_exams), 2) AS avg_exam_anxiety,
# MAGIC     ROUND(100.0 * AVG(strong_retention_flag), 2) AS strong_retention_pct
# MAGIC FROM ai_student_impact_clean
# MAGIC GROUP BY prompt_engineering_skill
# MAGIC ORDER BY avg_gpa_change DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Query 3: Institutional Policy and Burnout Risk

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     institutional_policy,
# MAGIC     COUNT(*) AS students,
# MAGIC     ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
# MAGIC     ROUND(AVG(skill_retention_score), 2) AS avg_skill_retention,
# MAGIC     ROUND(AVG(anxiety_level_during_exams), 2) AS avg_exam_anxiety,
# MAGIC     ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct,
# MAGIC     ROUND(100.0 * AVG(high_dependency_flag), 2) AS high_dependency_pct
# MAGIC FROM ai_student_impact_clean
# MAGIC GROUP BY institutional_policy
# MAGIC ORDER BY high_burnout_pct DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Query 4: Primary Use Case and Academic Outcomes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     primary_use_case,
# MAGIC     COUNT(*) AS students,
# MAGIC     ROUND(AVG(weekly_genai_hours), 2) AS avg_weekly_genai_hours,
# MAGIC     ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
# MAGIC     ROUND(AVG(skill_retention_score), 2) AS avg_skill_retention,
# MAGIC     ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct
# MAGIC FROM ai_student_impact_clean
# MAGIC GROUP BY primary_use_case
# MAGIC ORDER BY avg_gpa_change DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced SQL: Ranking AI Usage Bands Within Each Major
# MAGIC
# MAGIC This uses a window function to rank AI usage bands by average GPA improvement within each major.

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH band_stats AS (
# MAGIC     SELECT
# MAGIC         major_category,
# MAGIC         ai_usage_band,
# MAGIC         COUNT(*) AS students,
# MAGIC         ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
# MAGIC         ROUND(AVG(skill_retention_score), 2) AS avg_skill_retention
# MAGIC     FROM ai_student_impact_clean
# MAGIC     GROUP BY major_category, ai_usage_band
# MAGIC ),
# MAGIC ranked AS (
# MAGIC     SELECT
# MAGIC         *,
# MAGIC         RANK() OVER (
# MAGIC             PARTITION BY major_category
# MAGIC             ORDER BY avg_gpa_change DESC
# MAGIC         ) AS band_rank
# MAGIC     FROM band_stats
# MAGIC )
# MAGIC SELECT *
# MAGIC FROM ranked
# MAGIC WHERE band_rank <= 3
# MAGIC ORDER BY major_category, band_rank;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced SQL: Pivot GPA Change by Major and AI Usage Band

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC     SELECT major_category, ai_usage_band, gpa_change
# MAGIC     FROM ai_student_impact_clean
# MAGIC )
# MAGIC PIVOT (
# MAGIC     ROUND(AVG(gpa_change), 3)
# MAGIC     FOR ai_usage_band IN ('0-2 hours', '2-5 hours', '5-10 hours', '10-20 hours', '20+ hours')
# MAGIC )
# MAGIC ORDER BY major_category;

# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas Analysis

# COMMAND ----------

import pandas as pd

clean_pd = spark.table("ai_student_impact_clean").toPandas()

clean_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas Operation 1: GroupBy Summary by AI Usage Band

# COMMAND ----------

band_summary_pd = (
    clean_pd
    .groupby("ai_usage_band")
    .agg(
        students=("student_id", "count"),
        avg_gpa_change=("gpa_change", "mean"),
        avg_retention=("skill_retention_score", "mean"),
        avg_anxiety=("anxiety_level_during_exams", "mean"),
        high_burnout_pct=("high_burnout_flag", "mean")
    )
    .reset_index()
)

band_summary_pd["high_burnout_pct"] = band_summary_pd["high_burnout_pct"] * 100
display(band_summary_pd.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas Operation 2: Correlation Matrix
# MAGIC
# MAGIC This checks which numeric variables are most associated with GPA change and skill retention.

# COMMAND ----------

numeric_cols_for_corr = [
    "pre_semester_gpa",
    "weekly_genai_hours",
    "tool_diversity",
    "traditional_study_hours",
    "perceived_ai_dependency",
    "anxiety_level_during_exams",
    "post_semester_gpa",
    "skill_retention_score",
    "gpa_change",
    "high_burnout_flag"
]

corr_matrix = clean_pd[numeric_cols_for_corr].corr()
display(corr_matrix.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizations

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 1: Average GPA Change by AI Usage Band

# COMMAND ----------

band_chart = spark.sql("""
    SELECT
        ai_usage_band,
        ROUND(AVG(gpa_change), 3) AS avg_gpa_change,
        ROUND(AVG(skill_retention_score), 2) AS avg_retention,
        ROUND(100.0 * AVG(high_burnout_flag), 2) AS high_burnout_pct,
        COUNT(*) AS students
    FROM ai_student_impact_clean
    GROUP BY ai_usage_band
    ORDER BY
        CASE ai_usage_band
            WHEN '0-2 hours' THEN 1
            WHEN '2-5 hours' THEN 2
            WHEN '5-10 hours' THEN 3
            WHEN '10-20 hours' THEN 4
            WHEN '20+ hours' THEN 5
        END
""").toPandas()

fig1 = px.bar(
    band_chart,
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
fig1.update_traces(textposition="outside")
fig1.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 2: GenAI Hours vs Skill Retention

# COMMAND ----------

sample_pd = clean_pd.sample(n=min(5000, len(clean_pd)), random_state=42)

fig2 = px.scatter(
    sample_pd,
    x="weekly_genai_hours",
    y="skill_retention_score",
    color="burnout_risk_level",
    hover_data=["major_category", "year_of_study", "gpa_change"],
    title="Weekly GenAI Hours vs Skill Retention Score",
    labels={
        "weekly_genai_hours": "Weekly GenAI Hours",
        "skill_retention_score": "Skill Retention Score",
        "burnout_risk_level": "Burnout Risk"
    }
)
fig2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 3: Burnout Risk by AI Usage Band

# COMMAND ----------

burnout_chart = spark.sql("""
    SELECT
        ai_usage_band,
        burnout_risk_level,
        COUNT(*) AS students
    FROM ai_student_impact_clean
    GROUP BY ai_usage_band, burnout_risk_level
    ORDER BY ai_usage_band, burnout_risk_level
""").toPandas()

fig3 = px.bar(
    burnout_chart,
    x="ai_usage_band",
    y="students",
    color="burnout_risk_level",
    barmode="stack",
    title="Burnout Risk Distribution by AI Usage Band",
    labels={
        "ai_usage_band": "Weekly GenAI Usage Band",
        "students": "Number of Students",
        "burnout_risk_level": "Burnout Risk"
    }
)
fig3.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 4: Correlation Heatmap

# COMMAND ----------

fig4 = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Correlation Matrix for Student Outcome Variables"
)
fig4.show()


