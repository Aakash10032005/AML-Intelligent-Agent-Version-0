import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_aml_data(num_normal=5000, num_smurfs=50, num_layering=20):
    """
    Generates synthetic transaction data with normal behavior and injected 
    money laundering typologies:
    1. Structuring (Smurfing): Multiple transactions just under the $10,000 BSA threshold.
    2. Layering: Rapid pass-through transfers and fan-out to multiple accounts within short windows.
    """
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2025, 6, 25)
    
    # 1. Generate Normal Transactions
    normal_data = []
    for _ in range(num_normal):
        account_id = f"ACC_{random.randint(1000, 5000)}"
        amount = round(random.uniform(50, 3000), 2)
        timestamp = start_date + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        normal_data.append([account_id, timestamp, amount, "PAYMENT", 0, "Normal"])
        
    # 2. Inject Structuring (Smurfing) Patterns
    smurf_data = []
    for _ in range(num_smurfs):
        smurf_account = f"ACC_SMURF_{random.randint(9000, 9999)}"
        base_time = start_date + timedelta(days=random.randint(0, 28))
        
        num_transactions = random.randint(4, 6)
        for i in range(num_transactions):
            amount = round(random.uniform(9000, 9999.99), 2)
            timestamp = base_time + timedelta(hours=i*2) # 2 hours apart
            smurf_data.append([smurf_account, timestamp, amount, "TRANSFER", 1, "Structuring"])

    # 3. Inject Layering Patterns (Rapid Pass-Through / Fan-Out)
    layering_data = []
    for _ in range(num_layering):
        layer_account = f"ACC_LAYER_{random.randint(8000, 8999)}"
        base_time = start_date + timedelta(days=random.randint(0, 28))
        
        # Rapid pass-through: High volume incoming and immediate outbound transfers
        large_inflow = round(random.uniform(25000, 75000), 2)
        layering_data.append([layer_account, base_time, large_inflow, "TRANSFER", 1, "Layering"])
        
        # Fan-out into 3-5 split transactions within 1 hour
        splits = random.randint(3, 5)
        split_amount = round(large_inflow / splits, 2)
        for i in range(splits):
            timestamp = base_time + timedelta(minutes=(i + 1) * 10)
            layering_data.append([layer_account, timestamp, split_amount, "TRANSFER", 1, "Layering"])

    # 4. Combine, Format & Save
    all_data = normal_data + smurf_data + layering_data
    df = pd.DataFrame(all_data, columns=["account_id", "timestamp", "amount", "transaction_type", "is_suspicious", "pattern_type"])
    
    # Sort by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"[SUCCESS] Generated data/transactions.csv with {len(df)} records.")
    print(f"Included {len(smurf_data)} structuring transactions and {len(layering_data)} layering transactions.")

if __name__ == "__main__":
    generate_synthetic_aml_data()