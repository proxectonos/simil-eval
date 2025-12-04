import os
import re
import sys

import pandas as pd
import summarize_results
from excel_tools import collect_results_from_folder, prettify_excel, save_grouped_by_lang

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RESULTS_DIR = "/mnt/netapp1/Proxecto_NOS/adestramentos/simil-eval/logs_surprisal"
OUTPUT_DIR = "/mnt/netapp1/Proxecto_NOS/adestramentos/simil-eval/results_excel"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# PARSING LOGIC
# ----------------------------------------------------------------------
def parse_surprisal_results(text):
    """
    Parse surprisal benchmark results from a multi-language file.
    - If 'difsur-cola' present → Cola benchmark.
    - if 'difsur-globalpiqa present → Global Piqa benchmark.
    - If 'Mean score last word' present → Calame benchmark.
    Keeps only difsur and mean_last_word (removes good/bad mean).
    """
    results = []

    model_match = re.search(r"Results for model:\s*(.+)", text)
    model_path = model_match.group(1).strip() if model_match else "UNKNOWN"
    model = os.path.basename(model_path)

    # Find all blocks like "Launching surprisal test for X"
    blocks = re.findall(
        r"Launching surprisal\s+.*?\s+for\s+(.*?)[-]+\n([\s\S]*?)(?=########################################|$)",
        text
    )

    for lang, content in blocks:
        lang = lang.strip().lower()

        dif_cola_match = re.search(r"difsur-cola:\s*([0-9.]+)", content)
        dif_globalpiqa_match = re.search(r"difsur-globalpiqa:\s*([0-9.]+)", content)
        dif_match = dif_cola_match or dif_globalpiqa_match
        mean_match = re.search(r"Mean score last word:\s*([0-9.]+)", content)

        if dif_cola_match:
            benchmark = "cola"
            score = float(dif_cola_match.group(1))
        elif dif_globalpiqa_match:
            benchmark = "globalpiqa"
            score = float(dif_globalpiqa_match.group(1))
        elif mean_match:
            benchmark = "calame"
            score = float(mean_match.group(1))
        else:
            benchmark = "unknown"

        results.append({
            "model": model,
            "benchmark": benchmark,
            "lang": lang,
            "difsur": score if dif_match else None,
            "mean_last_word": score if mean_match else None,
        })

    return results


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = collect_results_from_folder(RESULTS_DIR, parse_surprisal_results, file_extension="out.log")

    if df.empty:
        print("⚠️ No results found.")
    else:
        excel_path = save_grouped_by_lang(df, OUTPUT_DIR, filename_prefix="results_surprisal")
        prettify_excel(excel_path)
        print(f"✅ Surprisal results exported and styled → {excel_path}")

        summarize_results.summarize_surprisal_from_path(OUTPUT_DIR)
