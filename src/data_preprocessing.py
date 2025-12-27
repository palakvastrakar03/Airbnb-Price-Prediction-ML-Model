import pandas as pd
import os

print(">>> data_preprocessing module loaded")

def preprocess_data(path):
    print(">>> preprocess_data function called")

    # Get project root (parent of src)
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Build absolute path to CSV
    full_path = os.path.join(project_root, path)

    print("Reading data from:", full_path)

    df = pd.read_csv(full_path)
    return df
