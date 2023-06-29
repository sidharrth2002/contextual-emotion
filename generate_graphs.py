import json
import pandas as pd
import os

all_results = {}

def clean_file_name(name):
    # remove the .json extension
    name = name.replace(".json", "")
    new_name = name.split("_")
    # remove "finetuned" from the name
    print(new_name)
    if "finetuned" in new_name:
        new_name.remove("finetuned")
    if "results" in new_name:
        new_name.remove("results")
    new_name = " + ".join(new_name)
    return new_name

for i in os.listdir("."):
    if i.endswith(".json"):
        with open(f"./{i}", "r") as f:
            data = json.load(f)
            all_results[clean_file_name(i)] = data

results = pd.DataFrame(all_results)
# switch rows and columns
# results = results.T
print(results)

results.to_csv("all_results_test_nonT.csv")