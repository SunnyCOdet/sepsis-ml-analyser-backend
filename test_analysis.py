import os
import sys
import json
from analysis import perform_analysis

# Add current directory to path
sys.path.append(os.getcwd())

# Path to the CSV file
csv_path = r"c:/Users/egnan/OneDrive/Desktop/janu/Sepsis data  5-Sheet 1-1-1-Sepsis excel sheetTable 1.csv"

if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    sys.exit(1)

try:
    print("Running analysis...")
    results = perform_analysis(csv_path)
    print("Analysis successful!")
    print(json.dumps(results, indent=2))
except Exception as e:
    print(f"Analysis failed: {e}")
    import traceback
    traceback.print_exc()
