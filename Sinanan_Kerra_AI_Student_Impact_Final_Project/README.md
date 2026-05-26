
# Kerra Sinanan Final Project: AI Student Impact Explorer

## Project Question

How does Generative AI use relate to student academic performance, skill retention, anxiety, and burnout risk?

## Dataset

File: `ai_student_impact_dataset.csv`

The dataset includes 50,000 student records with fields related to:

- Major category
- Year of study
- Pre-semester GPA
- Weekly GenAI hours
- Primary AI use case
- Prompt engineering skill
- Tool diversity
- Paid subscription status
- Traditional study hours
- Perceived AI dependency
- Institutional policy
- Exam anxiety
- Post-semester GPA
- Skill retention score
- Burnout risk level

## Main Cleaned Table

The notebook creates this table:

```sql
ai_student_impact_clean
```

The Streamlit app reads from that table.

## Files Included

| File | Purpose |
|---|---|
| `Sinanan_Kerra_AI_Student_Impact_Final_Project.py` | Databricks notebook source file |
| `app.py` | Streamlit Databricks app |
| `app.yaml` | Databricks app configuration |
| `requirements.txt` | Required Python packages |
| `ai_student_impact_dataset.csv` | Dataset to upload into Databricks |
| `slack_or_submission_reflection.txt` | Copy-ready reflection |
| `README.md` | Project documentation |

## Databricks Instructions


## Main Findings

1. Moderate GenAI use appears strongest. Students using GenAI around 5-10 hours per week had the highest average GPA improvement and strongest skill retention.
2. Heavy GenAI use is risky. Students using GenAI 20+ hours weekly had lower GPA improvement, lower retention, higher dependency, higher anxiety, and higher high-burnout rates.
3. Prompt engineering skill matters. Advanced prompt users had stronger GPA improvement and higher skill retention.
4. Direct answer generation was the weakest use case compared with more learning-centered uses like debugging, drafting, summarizing, and ideation.
5. The strongest interpretation is balance: GenAI works best as a study assistant, not as a replacement for learning.
