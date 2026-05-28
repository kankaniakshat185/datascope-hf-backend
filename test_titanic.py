import pandas as pd
from debugger import run_all_checks
from main import generate_data_dictionary, generate_eda_data, run_layer1_full
from layer1.services.shap_engine import compute_segmented_shap

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
target = "Survived"
print("Target column:", target)

issues = run_all_checks(df, target)
print("Run checks complete")

dict_data = generate_data_dictionary(df)
print("Dict complete")

eda_data = generate_eda_data(df)
print("EDA complete")

shap_raw = compute_segmented_shap(df, target)
print("SHAP complete")

try:
    l1 = run_layer1_full(df, target)
    print("Layer 1 complete")
except Exception as e:
    import traceback
    traceback.print_exc()

print("All done.")
