# Government-Schemes-Analysis-ml-

### 📌 Overview
Behind every government policy is a promise, and behind every budget is a vision for impact. This project performs an end-to-end **Exploratory Data Analysis (EDA)** on a comprehensive dataset of over 1,500 Indian Government Schemes. 

As a CSE student minoring in Data Science, my objective was to move beyond raw numbers and uncover the operational mechanics of public governance. The analysis measures budget efficiency, operational health across ministries, and the socio-economic reach of flagship programs.

### 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Environment:** Jupyter Notebook and VS Code

---

### 🧹 Data Quality & Preprocessing
Real-world policy data is notoriously messy. Before generating insights, a rigorous data cleaning pipeline was built to handle missing values without sacrificing statistical integrity:
* **Category-Specific Median Imputation:** Missing financial metrics (like `Budget_Allocated_Crore`) were imputed using the median of their specific sector (e.g., Housing, Sanitation) rather than the global average.
* **Logical Consistency Enforcement:** Built dynamic masks to catch and correct data entry anomalies (e.g., ensuring `Grievances_Resolved` never mathematically exceeded `Grievances_Filed`).
* **Text Standardization:** Handled nulls in categorical features via mode imputation and stripped leading/trailing whitespace for accurate grouping.

---

### 📊 Key Insights & Visualizations

**1. The Efficiency Benchmark (Scatter Plot)**
* Analyzed `Budget Allocated` vs. `Budget Utilized` across all schemes. By mapping a 100% utilization benchmark line, it became immediately clear which sectors suffer from operational bottlenecks and under-spending.

**2. Accountability in Flagship Programs (Violin Plot)**
* Compared the distribution of success rates between standard programs and 'PM Flagship' initiatives. The density curves revealed that flagship programs maintain a much tighter, higher distribution of on-the-ground success.

**3. The Digital Shift in Delivery (Pie Chart)**
* Mode distribution analysis showed a massive reliance on **Direct Benefit Transfers (DBT)**, highlighting the government's transition away from traditional reimbursement models toward digital financial inclusion.

**4. Multivariate Strategy Analysis (Pairplot)**
* Combined Success Rate, Utilization %, and Beneficiary Reach into a single KDE density grid, color-coded by delivery mode. This highlighted exactly which operational strategies scale most effectively.

---

🤖 Machine Learning: Predictive Policy Modeling
To transition from descriptive analytics to predictive analytics, two distinct ML pipelines were developed:

1. Success Rate Forecasting (Regression Model)
Objective: Predict the continuous Success_Rate_% of a given scheme.

Why it matters: Allows policymakers to run "what-if" scenarios. If we increase the budget by 10% and change the delivery mode to Digital Transfer, how will that impact the success rate?

Features Used: Budget_Allocated, Mode (Categorical), Target_Beneficiaries, Scheme_Duration_Years.

2. Operational Risk Flagging (Classification Model)
Objective: Classify the operational trajectory of a scheme (e.g., Predicting if a scheme is likely to remain Active or fall Under Review).

Why it matters: Acts as an early-warning system. By identifying patterns in historical data (such as high grievances coupled with low budget utilization), the model flags at-risk schemes for proactive administrative intervention.

Evaluation Metrics: Optimized for Recall to ensure failing schemes are accurately caught before resources are entirely depleted.

---

🎯 Conclusion & Future Scope 🚀

This project demonstrates that analyzing public policy requires more than just tallying budgets 💰; it requires uncovering the operational mechanics of governance. ⚙️🏛️ Through comprehensive Exploratory Data Analysis (EDA) 📊 and predictive Machine Learning 🤖, we identified clear patterns in how Indian Government Schemes succeed or fail. 🇮🇳 We discovered that Direct Benefit Transfers (DBT) 💸 are rapidly becoming the dominant, most efficient delivery mode 📱, and that 'PM Flagship' programs maintain significantly tighter accountability standards. 🛡️
---
