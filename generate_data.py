import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_aml_data(num_normal=5000, num_smurfs=50):
    """
    Generates synthetic transaction data with normal behavior and injected 
    structuring (smurfing) money laundering patterns.
    """
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    
    # 1. Generate Normal Transactions
    normal_data = []
    for _ in range(num_normal):
        account_id = f"ACC_{random.randint(1000, 5000)}"
        amount = round(random.uniform(50, 3000), 2)
        timestamp = start_date + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        normal_data.append([account_id, timestamp, amount, "PAYMENT", 0, "Normal"])
        
    # 2. Inject Structuring (Smurfing) Patterns
    # Smurfing: Multiple transactions just under the $10k reporting threshold within a short time.
    smurf_data = []
    for _ in range(num_smurfs):
        smurf_account = f"ACC_SMURF_{random.randint(9000, 9999)}"
        base_time = start_date + timedelta(days=random.randint(0, 28))
        
        # Create 4-6 rapid transactions just under $10,000 to evade reporting
        num_transactions = random.randint(4, 6)
        for i in range(num_transactions):
            amount = round(random.uniform(9000, 9999), 2)
            timestamp = base_time + timedelta(hours=i*2) # 2 hours apart
            smurf_data.append([smurf_account, timestamp, amount, "TRANSFER", 1, "Structuring"])

    # 3. Combine and Shuffle
    all_data = normal_data + smurf_data
    df = pd.DataFrame(all_data, columns=["account_id", "timestamp", "amount", "transaction_type", "is_suspicious", "pattern_type"])
    
    # Sort by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"✅ Generated data/transactions.csv with {len(df)} records.")
    print(f"Included {len(smurf_data)} structured transactions.")

if __name__ == "__main__":
    generate_synthetic_aml_data()