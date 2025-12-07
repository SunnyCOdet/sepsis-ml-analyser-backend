import sys
print("Python version:", sys.version)
print("Starting Flask app test...")

try:
    from flask import Flask
    print("✓ Flask imported successfully")
    
    from analysis import perform_analysis, predict_patient
    print("✓ Analysis functions imported successfully")
    
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    print("✓ All dependencies imported successfully")
    
    print("\nAll imports successful! Backend should work.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
