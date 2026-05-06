import sys
print("About to import RFXFuse...")
sys.stdout.flush()
import RFXFuse
print("RFXFuse imported successfully")
print(f"Has GPU support: {hasattr(RFXFuse, 'RandomForestClassifier')}")
