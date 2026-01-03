
import os
import sys
# Add current directory to path so we can plug in
sys.path.append(os.getcwd())

from advanced_analysis import perform_analysis

# Path to the CSV file
csv_path = r"c:\Users\egnan\OneDrive\Desktop\janu\backend\uploads\Sepsis data  5-Sheet 1-1-1-Sepsis excel sheetTable 1.csv"

if os.path.exists(csv_path):
    print(f"Analyzing {csv_path}...")
    results = perform_analysis(csv_path)
    print("\nCORRELATION RESULTS:")
    corr = results.get('correlation', {})
    for k, v in corr.items():
        print(f"{k}: {v}")
else:
    print("CSV file not found")
