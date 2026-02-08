# AWS-Legacy-Modernization
<img width="1907" height="857" alt="image" src="https://github.com/user-attachments/assets/a2e7a884-8f93-4ad1-8462-5c271c3b3122" />

Crawler-
<img width="1919" height="797" alt="image" src="https://github.com/user-attachments/assets/9c09216b-71be-4e1e-8a15-2f6ffa1f56ef" />

Athena-
Table
<img width="1919" height="822" alt="image" src="https://github.com/user-attachments/assets/906854dc-33f4-4d32-8eb8-fddc8f73f843" />


### 🏥 Healthcare Data Mastering & Ingestion Pipeline

This project is a personal end-to-end implementation of a healthcare data pipeline. I built this to solve a common industry challenge: handling inconsistent and "dirty" patient data coming from various sources. I used **AWS** and **PySpark** to automate the validation, cleaning, and mastering of these records.

#### 🛠️ Tech Stack & Tools

* **AWS Glue (PySpark):** The core engine used for heavy data processing and ETL logic. ⚙️
* **Amazon S3:** Used as a Data Lake with a layered architecture (Landing, Master, and Rejected zones). 🗄️
* **AWS Lambda:** Triggered on file upload to perform initial metadata and control-file validation. ⚡
* **Amazon Athena:** Used for serverless SQL queries to analyze processing results. 🔍
* **Amazon SES:** Configured to send a professional HTML status report directly to stakeholders after every job run. 📧

#### 🏗️ Key Features I Implemented

1. **Automated Quality Gate:** I set up a Lambda function to intercept files in the landing zone. Only files that pass the metadata check move forward, preventing "garbage data" from entering the lake.
2. **Data Mastering (MDM):** I used **Inner Joins** with a Prescriber Master list. This ensures that every patient record is mapped to a verified doctor (NPI validation).
3. **Deduplication (CDC):** To handle daily updates, I used **Spark Window Functions**. This logic identifies the most recent record for each patient, ensuring the master table stays accurate without duplicates.
4. **Intelligent Error Handling:** Records with missing IDs or invalid NPIs aren't just dropped; they are routed to a `rejected/` folder with specific error tags for auditing.

#### 📈 Execution Results

After the pipeline runs, it generates an automated HTML report. This gives me a high-level view of the data health immediately.

<img width="500" height="699" alt="image" src="https://github.com/user-attachments/assets/53580e23-4936-4ee6-8519-ab05e319f9e0" />

