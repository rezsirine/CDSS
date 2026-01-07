import sys
from pathlib import Path

# Simulate the Streamlit app path setup
sys.path.append(str(Path(__file__).parent))

try:
    from agents.symptom_agent import SymptomAgent
    agent = SymptomAgent(verbose=True)
    print(" Agent initialized successfully")
except Exception as e:
    print(f" Failed to initialize agent: {e}")
    import traceback
    traceback.print_exc()
