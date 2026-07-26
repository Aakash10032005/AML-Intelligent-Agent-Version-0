import sqlite3
import pandas as pd
import os

DB_FILE = "aml_transactions.db"
CSV_FILE = "transactions.csv"

def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    """Initializes SQLite, syncing from the CSV if the database table doesn't exist yet."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions';")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        if os.path.exists(CSV_FILE):
            print(f"[INFO] Importing {CSV_FILE} into SQLite database...")
            df_csv = pd.read_csv(CSV_FILE)
            df_csv.to_sql("transactions", conn, if_exists="replace", index=False)
            print("[INFO] Successfully imported CSV data!")
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    account_id TEXT,
                    timestamp TEXT,
                    amount REAL,
                    transaction_type TEXT,
                    is_suspicious INTEGER,
                    pattern_type TEXT
                )
            """)
            conn.commit()
            print(f"[WARNING] {CSV_FILE} not found. Created empty schema table.")
            
    conn.close()

def fetch_data(query="SELECT * FROM transactions"):
    conn = get_connection()
    try:
        df = pd.read_sql(query, conn)
    except Exception:
        df = pd.DataFrame(columns=["account_id", "timestamp", "amount", "transaction_type", "is_suspicious", "pattern_type"])
    conn.close()
    return df

def update_database_and_csv(df):
    """Pushes UI grid updates to SQLite AND synchronizes the physical CSV file."""
    conn = get_connection()
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    
    # Sync changes back to the physical CSV file
    df.to_csv(CSV_FILE, index=False)
    print(f"[INFO] Successfully synced changes back to {CSV_FILE}!")