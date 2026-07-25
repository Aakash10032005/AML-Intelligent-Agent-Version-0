import pandas as pd
from langchain_core.tools import tool
from sklearn.ensemble import IsolationForest

# Define data path
DATA_PATH = "data/transactions.csv"

@tool
def run_eda_tool(query: str = "all") -> str:
    """
    Use this tool to perform exploratory data analysis (EDA) on the transactions dataset to understand baseline behavior.
    Call this when the user asks for a general summary, broad exploration, or baseline statistics.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        total_tx = len(df)
        total_vol = df['amount'].sum()
        avg_amt = df['amount'].mean()
        suspicious_flags = df['is_suspicious'].sum()
        
        summary = (f"Exploratory Data Analysis Summary:\n"
                   f"- Total Transactions: {total_tx}\n"
                   f"- Total Volume: ${total_vol:,.2f}\n"
                   f"- Average Transaction Size: ${avg_amt:,.2f}\n"
                   f"- Baseline Suspicious Flags (from raw data): {suspicious_flags}")
        return summary
    except Exception as e:
        return f"Error running EDA: {str(e)}"

@tool
def detect_structuring_tool(timeframe: str = "30 days") -> str:
    """
    Use this tool specifically to detect anomalous transaction patterns indicative of money laundering, such as structuring or smurfing.
    Call this when the user asks to find structuring, smurfing, or multiple transactions just under reporting thresholds.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Rule-based detection: Find amounts between $9,000 and $9,999.99
        suspicious_amounts = df[(df['amount'] >= 9000) & (df['amount'] < 10000)]
        
        # Group by account to find repeat offenders
        counts = suspicious_amounts.groupby('account_id').size()
        smurfs = counts[counts >= 3].index.tolist()
        
        if not smurfs:
            return "No structuring patterns detected in the current dataset."
            
        return (f"Structuring Detected! The following accounts made 3 or more transactions "
                f"just under the $10,000 threshold, indicating potential smurfing: "
                f"{', '.join(smurfs[:5])} (Showing top 5 highest risk accounts).")
    except Exception as e:
        return f"Error detecting structuring: {str(e)}"

@tool
def risk_scoring_tool(account_id: str) -> str:
    """
    Use this tool to generate a risk score and classify a specific customer account using Machine Learning anomaly detection.
    Call this when the user asks "Is customer ID [X] suspicious?" or wants to check a specific entity.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        if account_id not in df['account_id'].values:
            return f"Account {account_id} not found in the dataset."
            
        # Feature Engineering for ML Model
        features = df.groupby('account_id').agg(
            tx_count=('amount', 'count'),
            avg_amount=('amount', 'mean'),
            max_amount=('amount', 'max')
        ).reset_index()
        
        # Train an Isolation Forest for Anomaly Detection
        model = IsolationForest(contamination=0.05, random_state=42)
        features['anomaly_score'] = model.fit_predict(features[['tx_count', 'avg_amount', 'max_amount']])
        
        # Lookup the requested account
        acc_data = features[features['account_id'] == account_id].iloc[0]
        
        # Convert model output to Risk Classification
        risk_level = "High" if acc_data['anomaly_score'] == -1 else "Low"
        
        return (f"Risk Assessment for {account_id}:\n"
                f"- Risk Classification: {risk_level} Risk\n"
                f"- Behavioral Profile: {acc_data['tx_count']} total transactions, Average Amount: ${acc_data['avg_amount']:,.2f}, Max Amount: ${acc_data['max_amount']:,.2f}.")
    except Exception as e:
        return f"Error scoring account: {str(e)}"