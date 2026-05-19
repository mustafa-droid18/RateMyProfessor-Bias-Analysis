# RateMyProfessor-Bias-Analysis

Capstone project for *Introduction to Data Science (DS GA 1001)*, analyzing gender bias and perception trends in professor ratings using data from RateMyProfessor.com.

## 👥 Team
- Mustafa Poonawala (msp9471)
- Aysha Allahverdiyeva (aa7983)

---

## 🧠 Project Summary

This project investigates **gender bias**, **perceived difficulty**, and **tag-driven stereotypes** in student evaluations of professors. Using a large dataset scraped from RateMyProfessor.com, we applied Bayesian adjustments, statistical hypothesis testing, and visualizations to uncover patterns in how students rate and describe faculty.

---

## 📌 Key Questions & Findings

| **Question** | **Summary of Findings** |
|-------------|--------------------------|
| **Q1:** Is there pro-male bias in professor ratings? | Yes, male professors received slightly higher ratings (mean diff = 0.03, p < 0.005). Bias persists across subsets. |
| **Q2:** Is rating variance different between genders? | No. Variances were similar. Levene’s test not significant (p = 0.0082 > 0.005). |
| **Q3:** What’s the size of the bias? | Small but statistically significant. Cliff’s delta = 0.0386 (negligible effect). |
| **Q4:** Are certain tags gendered? | Yes. 18/20 tags showed significant gender differences. “Hilarious” and “Amazing Lectures” for males; “Participation Matters” for females. |
| **Q5:** Is there gender bias in difficulty ratings? | Slight bias found. Male professors perceived as less difficult. Difference was statistically significant but small. |
| **Q6:** Is there a difference in perceived quality between online and offline classes? | Yes. Offline classes received significantly higher ratings than online ones (mean diff ≈ 0.26). Ratings were adjusted using Bayesian smoothing. |
| **Q7:** Are “pepper” professors rated more favorably, and does this vary by gender? | Yes. “Pepper” professors had higher average ratings. This effect was stronger for male professors. Gender interacted with “pepper” status in perceived quality. |
| **Q8:** Do highly rated professors tend to be viewed as easier or harder? | Higher-rated professors tended to be rated as easier. Negative correlation between average quality and difficulty. Bias toward favoring “easier” professors. |
| **Q9:** Is there a difference in ratings by field (major)? | Yes. Professors in Humanities and Arts fields generally received higher ratings. Quantitative fields (e.g., Math, Engineering) saw lower average ratings. |
| **Q10:** Are there university-level differences in professor ratings? | Yes. Significant variation across universities. Some universities consistently had higher or lower-rated professors. Regional patterns were also observed. |

---
## Data Files

- `rmpCapstoneNum.csv`: Numeric professor ratings and metadata
- `rmpCapstoneQual.csv`: Qualitative data (field, university, state)
- `rmpCapstoneTags.csv`: Frequency of tags assigned to professors

All files include 89,893 entries representing individual professors.

## 📁 Project Structure
```
📦 Assessing-Bias-Professor-Ratings
├── data/
│   ├── rmpCapstoneNum.csv
│   ├── rmpCapstoneQual.csv
│   └── rmpCapstoneTags.csv
├── src/
│   └── IDS_Capstone_Project_Final.py
├── reports/
│   ├── IDS Capstone Project Report.pdf
│   └── IDS capstone project spec sheet.pdf
├── README.md
```


---

## ⚙️ How to Run

1. Install dependencies listed below.
2. Run the `IDS_Capstone_Project_Final.py` script to reproduce all analyses and visualizations.
3. Optional: explore the preprocessing workflow in `Preproceesing.ipynb`.

---

## 📦 Requirements

```bash
pandas
numpy
matplotlib
scikit-learn
scipy
statsmodels
imbalanced-learn
```
## 📄 Final Project Report
[Final Report (PDF)](./reports/IDS%20Capstone%20Project%20Report.pdf)

