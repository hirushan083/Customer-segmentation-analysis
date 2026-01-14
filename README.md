# Customer-segmentation-analysis

Customer Segmentation Analysis 📊

Developed a machine learning model using K-Means and DBSCAN to group retail customers by income and spending habits, achieving a 0.54 Silhouette Score and utilizing PCA for 2D visualization.

🚀 Project Overview

This project uses the "Mall Customers" dataset to perform clustering analysis. By identifying specific consumer segments, businesses can tailor their services and promotions to high-value customers or those with specific spending habits.

Key Objectives:

Perform exploratory data analysis (EDA) to understand customer distributions.
Apply K-Means and DBSCAN clustering to find natural groupings.
Evaluate model performance using Silhouette Scores and Davies-Bouldin Index.
Visualize high-dimensional data using PCA (Principal Component Analysis)

🛠️ Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

Environment: Jupyter Notebook / Streamlit (for deployment)

📈 Key Results
The analysis successfully identified five primary customer segments:

High Earners, High Spenders: Target for luxury rewards.
High Earners, Low Spenders: Target for high-end promotional offers.
Average Earners, Average Spenders: The stable "standard" customer.
Low Earners, High Spenders: Often younger consumers or trend-driven.
Low Earners, Low Spenders: Price-sensitive customers.

💻 How to Run
Clone the repository

git clone https://github.com/hirushan083/Customer-segmentation-analysis.git

Install dependencies

pip install -r requirements.txt

Run on cmd

streamlit run app.py

✍️ Author
**Kavindu Hirushan**
* GitHub: [@hirushan083](https://github.com/hirushan083)
* Email: kavinduhirushan083@gmail.com
* 
📂 Dataset Source
The dataset used in this project is the **Mall Customer Segmentation Data**, which is a popular dataset for learning unsupervised machine learning concepts.

* **Source:** [Kaggle - Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
* **Description:** Contains 200 records of customer demographics (Age, Gender) and financial data (Annual Income, Spending Score).
