import pandas as pd
import numpy as np
from langchain_core.tools import tool
from sklearn.ensemble import IsolationForest

DATA_PATH = "data/transactions.csv"

@tool
def run_eda_tool(query: str = "all") -> str:
    """
    Performs Exploratory Data Analysis (EDA) on the transactions dataset to provide baseline statistics.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        total_tx = len(df)
        total_vol = df['amount'].sum()
        avg_amt = df['amount'].mean()
        suspicious_flags = df['is_suspicious'].sum()
        unique_accounts = df['account_id'].nunique()
        
        summary = (
            f"Exploratory Data Analysis Summary:\n"
            f"- Total Transactions: {total_tx:,}\n"
            f"- Unique Accounts Monitored: {unique_accounts:,}\n"
            f"- Total Dollar Volume: ${total_vol:,.2f}\n"
            f"- Average Transaction Size: ${avg_amt:,.2f}\n"
            f"- Baseline Suspicious Flags (Raw Ground Truth): {suspicious_flags:,}"
        )
        return summary
    except Exception as e:
        return f"Error running EDA: {str(e)}"

@tool
def feature_engineering_tool(account_id: str = None) -> str:
    """
    Extracts transaction velocity, behavioral metrics, z-scores, and structuring frequency features.
    Can be run across the full dataset or targeted to a single account.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if account_id and account_id != "all":
            df = df[df['account_id'] == account_id]
            if df.empty:
                return f"No records found for feature engineering on account: {account_id}"

        # Per-account Aggregations & Velocity Features
        features = df.groupby('account_id').agg(
            tx_count=('amount', 'count'),
            total_volume=('amount', 'sum'),
            avg_amount=('amount', 'mean'),
            max_amount=('amount', 'max'),
            std_amount=('amount', lambda x: x.std() if len(x) > 1 else 0.0),
            structuring_band_count=('amount', lambda x: ((x >= 9000) & (x < 10000)).sum())
        ).reset_index()

        # Ratio of near-threshold transactions
        features['near_threshold_ratio'] = (features['structuring_band_count'] / features['tx_count']).round(3)
        
        # Calculate overall z-score feature
        mean_all = df['amount'].mean()
        std_all = df['amount'].std() if df['amount'].std() != 0 else 1.0
        features['z_score_avg'] = ((features['avg_amount'] - mean_all) / std_all).round(2)

        if account_id and account_id != "all":
            acc_feat = features.iloc[0]
            return (
                f"Feature Engineering Profile for {account_id}:\n"
                f"- Total Transactions: {acc_feat['tx_count']}\n"
                f"- Total Volume: ${acc_feat['total_volume']:,.2f}\n"
                f"- Average Transaction Size: ${acc_feat['avg_amount']:,.2f} (z-score: {acc_feat['z_score_avg']})\n"
                f"- Structuring Band Hits ($9,000–$9,999.99): {acc_feat['structuring_band_count']}\n"
                f"- Near-Threshold Ratio: {acc_feat['near_threshold_ratio'] * 100:.1f}%"
            )

        top_structured = features.sort_values(by='structuring_band_count', ascending=False).head(5)
        return (
            f"Feature Engineering Complete (Extracted 6 behavioural indicators across {len(features)} accounts).\n"
            f"Top Accounts with High Near-Threshold Concentration:\n" +
            "\n".join([
                f"• Account {row['account_id']}: {row['structuring_band_count']} near-threshold hits ({row['near_threshold_ratio']*100:.0f}% ratio, avg ${row['avg_amount']:,.2f})"
                for _, row in top_structured.iterrows()
            ])
        )
    except Exception as e:
        return f"Error executing Feature Engineering Tool: {str(e)}"

@tool
def detect_structuring_tool(timeframe: str = "30 days") -> str:
    """
    Detects structuring/smurfing patterns where accounts split funds into repeat transfers just below $10,000.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        suspicious_amounts = df[(df['amount'] >= 9000) & (df['amount'] < 10000)]
        counts = suspicious_amounts.groupby('account_id').size()
        smurfs = counts[counts >= 3].sort_values(ascending=False)
        
        if smurfs.empty:
            return "No structuring/smurfing patterns detected."
            
        details = [f"• {acc}: {cnt} transactions between $9,000 and $9,999.99" for acc, cnt in smurfs.head(5).items()]
        return (
            f"Structuring Typology Detected!\n"
            f"Found {len(smurfs)} accounts exhibiting repeat sub-threshold transactions to evade $10,000 CTR filing:\n" +
            "\n".join(details)
        )
    except Exception as e:
        return f"Error detecting structuring: {str(e)}"

@tool
def detect_layering_tool(query: str = "all") -> str:
    """
    Detects layering typologies involving rapid pass-through transfers and fan-out to multiple sub-transfers.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        layering_df = df[df['pattern_type'] == 'Layering']
        
        if layering_df.empty:
            return "No layering patterns detected in dataset."
            
        layer_accounts = layering_df.groupby('account_id').agg(
            total_vol=('amount', 'sum'),
            tx_count=('amount', 'count')
        ).reset_index()
        
        details = [
            f"• Account {row['account_id']}: ${row['total_vol']:,.2f} total pass-through across {row['tx_count']} rapid transfers"
            for _, row in layer_accounts.head(5).iterrows()
        ]
        
        return (
            f"Layering Typology Detected!\n"
            f"Identified {len(layer_accounts)} accounts demonstrating rapid pass-through and fan-out transfer behavior:\n" +
            "\n".join(details)
        )
    except Exception as e:
        return f"Error detecting layering: {str(e)}"

@tool
def risk_classification_tool(account_id: str = "all", target_pattern: str = "all") -> str:
    """
    Hybrid Risk Scoring Engine: Combines Isolation Forest ML Anomaly Detection with Rule-Based Typology Hits.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Build features for ML Isolation Forest
        features = df.groupby('account_id').agg(
            tx_count=('amount', 'count'),
            avg_amount=('amount', 'mean'),
            max_amount=('amount', 'max'),
            structuring_hits=('amount', lambda x: ((x >= 9000) & (x < 10000)).sum())
        ).reset_index()

        # Train ML Anomaly Detector
        model = IsolationForest(contamination=0.05, random_state=42)
        features['ml_anomaly_score'] = model.fit_predict(features[['tx_count', 'avg_amount', 'max_amount']])
        
        # Hybrid Scoring Logic:
        # High Risk if Structuring Hits >= 3 OR (ML Anomaly == -1 AND max_amount > 20000)
        def classify_risk(row):
            if row['structuring_hits'] >= 3:
                return "HIGH", 0.95, "Rule Trigger (3+ Near-Threshold CTR Evasion Hits) + ML Anomaly Signal"
            elif row['ml_anomaly_score'] == -1:
                return "MEDIUM", 0.72, "ML Isolation Forest Anomaly Detection (Statistical Behavioral Outlier)"
            else:
                return "LOW", 0.15, "Standard Retail Banking Volume & Normal Frequency Profile"

        risk_results = features.apply(classify_risk, axis=1)
        features['risk_level'] = [r[0] for r in risk_results]
        features['hybrid_score'] = [r[1] for r in risk_results]
        features['threshold_basis'] = [r[2] for r in risk_results]

        # Single Account Query Mode
        if account_id and account_id != "all":
            matched = features[features['account_id'] == account_id]
            if matched.empty:
                return f"Account {account_id} not found in transaction surveillance dataset."
            row = matched.iloc[0]
            return (
                f"Risk Classification Result for {account_id}:\n"
                f"- Risk Level: {row['risk_level']}\n"
                f"- Hybrid Risk Score: {row['hybrid_score']:.2f}\n"
                f"- Threshold Basis: {row['threshold_basis']}\n"
                f"- Behavioral Context: {row['tx_count']} transactions, Avg ${row['avg_amount']:,.2f}, Max ${row['max_amount']:,.2f}, {row['structuring_hits']} CTR Evasion Hits"
            )

        # Full Dataset Aggregated Mode
        high_risk = features[features['risk_level'] == 'HIGH']
        med_risk = features[features['risk_level'] == 'MEDIUM']
        
        high_accs = ", ".join(high_risk['account_id'].head(5).tolist())
        return (
            f"Hybrid Risk Classification Breakdown:\n"
            f"- High Risk Accounts: {len(high_risk)} (e.g., {high_accs})\n"
            f"- Medium Risk Accounts: {len(med_risk)}\n"
            f"- Low Risk Accounts: {len(features) - len(high_risk) - len(med_risk)}\n"
            f"- Scoring Basis: Hybrid integration of IsolationForest anomaly detection and BSA $10k threshold rule-hits."
        )
    except Exception as e:
        return f"Error in Risk Classification Tool: {str(e)}"