# AI-Powered Suspicious Activity Detection Agent

An intelligent, autonomous agent designed to identify anomalous transaction patterns and provide explainable risk assessments.

## Dataset Information & Generation Logic
In accordance with the hackathon rules, this project utilizes a custom synthetic dataset generated via Python (`generate_data.py`). 

**Schema & Field Definitions:**
* `account_id` (String): Unique identifier for the customer account.
* `timestamp` (Datetime): Time of the transaction.
* `amount` (Float): Transaction value in USD.
* `transaction_type` (String): Type of movement (e.g., PAYMENT, TRANSFER).
* `is_suspicious` (Integer): Binary flag (1 = Suspicious, 0 = Normal) used for model evaluation.
* `pattern_type` (String): The specific typology (e.g., Normal, Structuring).

**Assumptions & Logic:**
The data simulates standard retail banking behavior mixed with specific Anti-Money Laundering (AML) typologies. Specifically, it injects "Structuring/Smurfing" patterns where malicious actors execute 4-6 rapid transfers just under the $10,000 reporting threshold (e.g., $9,000 - $9,999) within a 24-hour window to evade detection.