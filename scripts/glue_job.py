import sys
import boto3
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.transforms import *
from pyspark.sql.functions import col, upper, to_date, row_number, current_timestamp, lit
from pyspark.sql import Window
from botocore.exceptions import ClientError

# ==============================================================================
# 1. INITIALIZATION & CONFIGURATION
# ==============================================================================
# Initializing Glue Context and Spark Session
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# S3 Paths for Data Lake Layers
input_path = "s3://prashant-aws-learning-2026/landing/NVS_LEAP_CASE_20260206.txt"
output_path = "s3://prashant-aws-learning-2026/master/"
config_path = "s3://prashant-aws-learning-2026/config/prescriber_master.txt"
reject_base_path = "s3://prashant-aws-learning-2026/rejected/"

# ==============================================================================
# 2. DATA INGESTION & STANDARDIZATION
# ==============================================================================
# Reading Raw Pipe-Separated File from Landing Zone
df = spark.read.option("header", "true").option("sep", "|").csv(input_path)

# Data Standardization: Converting to Uppercase and Correcting Date Formats
# This ensures consistency across the data lake
standardized_df = df.withColumn("patient_name", upper(col("patient_name"))) \
                    .withColumn("state", upper(col("state"))) \
                    .withColumn("dob", to_date(col("dob"), "yyyy-MM-dd")) \
                    .withColumn("gender", upper(col("gender")))

# Adding Load Timestamp to track when the data was processed (Critical for CDC)
standardized_delta = standardized_df.withColumn("load_timestamp", current_timestamp())

# ==============================================================================
# 3. DATA QUALITY & MASTERING LOGIC
# ==============================================================================
# Step A: Filter records where Primary Key (spp_pat_id) is present
success_df = standardized_delta.filter(col("spp_pat_id").isNotNull())

# Step B: Prescriber (Doctor) Master Lookup
# Reading reference data to validate if the doctor associated with the patient is valid
pres_master_df = spark.read.option("header", "true").option("sep", "|").csv(config_path)

# Inner Join: Only keep records that have a matching valid NPI in the master list
mastered_df = success_df.join(pres_master_df, "npi_id", "inner")

# Step C: Capture Rejected Records
# 1. Invalid Doctor NPIs (Records that failed the join)
invalid_doctor_df = success_df.join(pres_master_df, "npi_id", "left_anti") \
                                .withColumn("error_message", lit("INVALID_PRESCRIBER_NPI"))

# 2. Missing Patient IDs (Records that failed the initial null check)
rejected_df = standardized_delta.filter(col("spp_pat_id").isNull()) \
                                .withColumn("error_message", lit("MISSING_PATIENT_ID"))

# Union: Combining all rejected records into a single DataFrame for auditing
final_rejected_df = rejected_df.unionByName(invalid_doctor_df)

# ==============================================================================
# 4. AUDIT COUNTS & REJECTS STORAGE
# ==============================================================================
total_count = standardized_delta.count()
success_count = mastered_df.count()
failed_count = final_rejected_df.count()

# Persisting Rejected Records to S3 in Parquet format for Athena Analysis
if failed_count > 0:
    final_rejected_df.write.mode("append").parquet(reject_base_path)

# ==============================================================================
# 5. CHANGE DATA CAPTURE (CDC) / UPSERT LOGIC
# ==============================================================================
# Using Window functions to pick the latest record per Patient ID to avoid duplicates
try: 
    # Read existing Master data
    master_df = spark.read.parquet(output_path) 
    # Combine existing data with new daily clean data
    combined_df = master_df.unionByName(mastered_df) 
    
    # Define Window: Partition by ID, sort by latest timestamp
    window_spec = Window.partitionBy("spp_pat_id").orderBy(col("load_timestamp").desc()) 
    
    # Filter Row Number 1 (Latest Record)
    final_df = combined_df.withColumn("rn", row_number().over(window_spec)) \
                          .filter(col("rn") == 1).drop("rn") 
    print("CDC Processed: Delta Merged Successfully") 
except Exception as e: 
    # If Master doesn't exist (Initial Load), use current clean data
    final_df = mastered_df 
    print(f"Initial Load or Error encountered: {str(e)}")

# Writing final deduplicated data back to Master Zone
final_df.write.mode("overwrite").parquet(output_path)

# ==============================================================================
# 6. AUTOMATED HTML REPORTING VIA AWS SES
# ==============================================================================
html_report = f"""
<html>
<head>
<style>
  table {{ font-family: Arial, sans-serif; border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
  tr:nth-child(even) {{ background-color: #f2f2f2; }}
  .header {{ background-color: #004466; color: white; padding: 10px; text-align: center; }}
  .status-warn {{ color: #cc0000; font-weight: bold; }}
  .status-ok {{ color: #008000; font-weight: bold; }}
</style>
</head>
<body>
  <div class="header">
    <h2>Healthcare ETL Processing Report</h2>
  </div>
  <p>Hello Team, the data ingestion for <b>NVS_LEAP_CASE</b> has completed.</p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Records Processed</td><td>{total_count}</td></tr>
    <tr><td>Successfully Mastered</td><td>{success_count}</td></tr>
    <tr class="{'status-warn' if failed_count > 0 else 'status-ok'}">
        <td>Rejected Records (Action Required)</td><td>{failed_count}</td></tr>
  </table>
  <p><b>S3 Rejects Path:</b> <a href="https://s3.console.aws.amazon.com/s3/buckets/prashant-aws-learning-2026?prefix=rejected/">View Rejects</a></p>
  <p><i>Note: Please query Athena 'rejected_records' table for error details.</i></p>
</body>
</html>
"""

# Initialize SES Client for communication
ses = boto3.client('ses', region_name='ap-south-1')

try:
    response = ses.send_email(
        Source='prashants9291@gmail.com',
        Destination={'ToAddresses': ['prashants9291@gmail.com']},
        Message={
            'Subject': {'Data': '🚀 Production Report: Healthcare Data Ingestion'},
            'Body': {'Html': {'Data': html_report}}
        }
    )
    print("✅ Professional HTML Email Sent Successfully!")
except ClientError as e:
    print(f"❌ SES Error: {e.response['Error']['Message']}")

print("✅ Glue Job completed successfully. Master Data Catalog Updated!")
