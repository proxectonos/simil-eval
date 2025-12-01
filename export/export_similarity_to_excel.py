import os
import re

import pandas as pd
import summarize_results
from excel_tools import prettify_excel
from tqdm.auto import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RESULTS_DIR = "/mnt/netapp1/Proxecto_NOS/adestramentos/simil-eval/logs_similarity"
OUTPUT_DIR = "/mnt/netapp1/Proxecto_NOS/adestramentos/simil-eval/results_excel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["openbookqa", "xstorycloze", "truthfulqa", "belebele", "veritasqa"]
LANGS = ["gl", "cat", "es", "en", "pt"]


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def parse_filename(filename):
    """
    Parse a filename in the format "date_model_dataset_lang.txt"
    into its constituent parts.

    Returns a tuple of (date, model, dataset, lang) if the filename is valid,
    or None otherwise.
    """
    name = filename.replace(".txt", "")
    parts = name.split("_")

    dataset_idx = next((i for i, p in enumerate(parts) if p in DATASETS), None)
    lang_idx = next((i for i, p in enumerate(parts) if p in LANGS), None)
    if dataset_idx is None or lang_idx is None:
        return None

    date = parts[1]
    model = "_".join(parts[2:dataset_idx])
    dataset = parts[dataset_idx]
    lang = parts[lang_idx]
    return date, model, dataset, lang


def parse_metrics(text):
    #"""Extract COSINE, MOVERSCORE, and BERTSCORE (only F1) metrics."""
    """
    Parse a text containing the results of a similarity evaluation and extract
    the following metrics:

    - COSINE: Mean similarity score, mean similarity score with correct options,
      and percentage of correct answers (over 1)
    - MOVERSCORE: Mean similarity score, mean similarity score with correct options,
      and percentage of correct answers (over 1)
    - BERTSCORE: F1 score for similarity with correct options, and F1 score for
      similarity with all options

    Returns a dictionary with the extracted metrics.
    """
    metrics = {}

    # COSINE
    cosine_match = re.search(r"--COSINE RESULTS--([\s\S]*?)(?=--MOVERSCORE RESULTS--|--BERTSCORE RESULTS--|$)", text)
    if cosine_match:
        block = cosine_match.group(1)
        metrics["cosine_mean"] = float(re.search(r"Global Mean similarity score:\s*([0-9.]+)", block).group(1))
        metrics["cosine_correct"] = float(re.search(r"Global Mean similarity score with correct options:\s*([0-9.]+)", block).group(1))
        metrics["cosine_acc"] = float(re.search(r"Percentage of correct answers \(over 1\):\s*([0-9.]+)", block).group(1))

    # MOVERSCORE
    mover_match = re.search(r"--MOVERSCORE RESULTS--([\s\S]*?)(?=--BERTSCORE RESULTS--|$)", text)
    if mover_match:
        block = mover_match.group(1)
        metrics["mover_mean"] = float(re.search(r"Global Mean similarity score:\s*([0-9.]+)", block).group(1))
        metrics["mover_correct"] = float(re.search(r"Global Mean similarity score with correct options:\s*([0-9.]+)", block).group(1))
        metrics["mover_acc"] = float(re.search(r"Percentage of correct answers \(over 1\):\s*([0-9.]+)", block).group(1))

    # BERTSCORE (F1 only)
    bert_match = re.search(r"--BERTSCORE RESULTS--([\s\S]*?)[-]{5,}\n?$", text)
    if bert_match:
        block = bert_match.group(1)
        correct_f1 = re.search(r"Similarity with correct options:\s*\[.*?f1:\s*([0-9.]+)", block, re.DOTALL)
        all_f1 = re.search(r"Similarity with all options:\s*\[.*?f1:\s*([0-9.]+)", block, re.DOTALL)
        if correct_f1:
            metrics["bert_correct_f1"] = float(correct_f1.group(1))
        if all_f1:
            metrics["bert_all_f1"] = float(all_f1.group(1))
    else:
        print("⚠️ BERTSCORE block not found in text")

    return metrics


def update_excel(lang, dataset, model, date, metrics):
    #"""Insert or update model results in the appropriate Excel sheet."""
    """
    Insert or update model results in the appropriate Excel sheet.

    Parameters
    ----------
    lang : str
        Language code (e.g. "en", "es", "gl")
    dataset : str
        Dataset name (e.g. "openbookqa", "xstorycloze", "truthfulqa")
    model : str
        Model name (e.g. "distilbert-base-uncased")
    date : str
        Date of the evaluation in the format "YYYY-MM-DD"
    metrics : dict
        Dictionary with evaluation metrics (COSINE, MOVERSCORE, BERTSCORE)

    Returns
    -------
    None

    Notes
    -----
    Results are sorted by model name in ascending order.
    Existing results are overwritten if the model name is the same.
    """
    excel_path = os.path.join(OUTPUT_DIR, f"results_{lang}.xlsx")
    sheet_name = dataset

    if os.path.exists(excel_path):
        with pd.ExcelFile(excel_path) as xls:
            df = pd.read_excel(xls, sheet_name=sheet_name) if sheet_name in xls.sheet_names else pd.DataFrame()
    else:
        df = pd.DataFrame()

    new_row = {"Model": model, "Date": date}
    new_row.update(metrics)

    # Remove old entry if exists
    if not df.empty:
        df = df[df["Model"] != model]

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values(by="Model", key=lambda col: col.str.lower()).reset_index(drop=True)

    # Order metrics consistently
    metric_order = [
        "cosine_acc", "cosine_mean", "cosine_correct",
        "mover_acc", "mover_mean", "mover_correct",
        "bert_correct_f1", "bert_all_f1"
    ]
    base_cols = ["Model", "Date"]
    ordered_cols = base_cols + [c for c in metric_order if c in df.columns] + [
        c for c in df.columns if c not in base_cols + metric_order
    ]
    df = df[ordered_cols]

    # Write to Excel
    mode = "a" if os.path.exists(excel_path) else "w"
    # FIXME: if_sheet_exists is only valid in append mode (crash if mode=="w"). Until further info, temporary fixing with an if-else case.
    # I'm guessing this should be easily fixed by setting mode to "a" always, but I don't want to change the logic too much without knowing the full context.
    if mode == "w":
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode=mode) as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    # Pretty formatting
    prettify_excel(excel_path)


def process_all():
    #"""Iterate over result files and build per-language Excels."""
    """
    Iterate over result files and build per-language Excels.

    This function processes all files in the RESULTS_DIR directory
    whose names start with "similarity_" and end with "_out.log".
    It parses the filename to extract the date, model name, dataset name,
    and language code. It then reads the file, extracts the evaluation metrics
    using parse_metrics, and updates the Excel sheet for the language using
    update_excel.

    After all files have been processed, it prints a success message.
    """
    for fname in tqdm(os.listdir(RESULTS_DIR), desc="Processing similarity results"):
        if not fname.startswith("similarity_") or not fname.endswith("_out.log"):
            continue

        parsed = parse_filename(fname)
        if not parsed:
            continue

        date, model, dataset, lang = parsed
        with open(os.path.join(RESULTS_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()

        metrics = parse_metrics(text)
        update_excel(lang, dataset, model, date, metrics)

    print("✅ All similarity results processed and styled.")

    summarize_results.summarize_similarity_from_path(OUTPUT_DIR)

if __name__ == "__main__":
    process_all()
