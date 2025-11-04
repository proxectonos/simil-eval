import os
import re
import pandas as pd

# Folder where your results are stored
RESULTS_DIR = "/mnt/netapp1/Proxecto_NOS/adestramentos/simil-eval/logs_similarity"

# Output folder for Excel files
OUTPUT_DIR = "/mnt/netapp1/Proxecto_NOS/adestramentos/simil-eval/results_excel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex to parse filenames
# similarity_{date}_{model}_{dataset}_{lang}_{other}
DATASETS = ["openbookqa", "xstorycloze", "truthfulqa", "belebele", "veritasqa"]
LANGS = ["gl", "cat", "es", "en", "pt"]

def parse_filename(filename):
    name = filename.replace(".txt", "")
    parts = name.split("_")
    # Find dataset and lang positions
    dataset_idx = next((i for i, p in enumerate(parts) if p in DATASETS), None)
    lang_idx = next((i for i, p in enumerate(parts) if p in LANGS), None)
    if dataset_idx is None or lang_idx is None:
        return None  # skip malformed names
    date = parts[1]
    model = "_".join(parts[2:dataset_idx])
    dataset = parts[dataset_idx]
    lang = parts[lang_idx]
    return date, model, dataset, lang


def parse_metrics(text):
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

    # BERTSCORE (only F1)
    bert_match = re.search(r"--BERTSCORE RESULTS--([\s\S]*?)[-]{5,}\n?$", text)
    if bert_match:
        block = bert_match.group(1)

        correct_f1 = re.search(
            r"Similarity with correct options:\s*\[.*?f1:\s*([0-9.]+)",
            block, re.DOTALL)
        all_f1 = re.search(
            r"Similarity with all options:\s*\[.*?f1:\s*([0-9.]+)",
            block, re.DOTALL)

        if correct_f1:
            metrics["bert_correct_f1"] = float(correct_f1.group(1))
        if all_f1:
            metrics["bert_all_f1"] = float(all_f1.group(1))
    else:
        print("⚠️ BERTSCORE block not found in text")

    return metrics


def update_excel(lang, dataset, model, date, metrics):
    excel_path = os.path.join(OUTPUT_DIR, f"results_{lang}.xlsx")
    sheet_name = dataset

    # Load existing data if exists
    if os.path.exists(excel_path):
        with pd.ExcelFile(excel_path) as xls:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            else:
                df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Create new row
    new_row = {"Model": model, "Date": date}
    new_row.update(metrics)

    # Replace row for same model/date combination or append new
    if not df.empty:
        # If model already exists, replace it
        df = df[df["Model"] != model]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Order models consistently (alphabetically) ---
    df = df.sort_values(by="Model", key=lambda col: col.str.lower()).reset_index(drop=True)

    # --- Order metric columns so "_acc" appears first per metric type ---
    metric_order = [
        # COSINE
        "cosine_acc", "cosine_mean", "cosine_correct",
        # MOVER
        "mover_acc", "mover_mean", "mover_correct",
        # BERT
        "bert_correct_f1", "bert_all_f1"
    ]

    # Reorder columns dynamically (preserving any new metrics added later)
    base_cols = ["Model", "Date"]
    ordered_cols = base_cols + [c for c in metric_order if c in df.columns] + [
        c for c in df.columns if c not in base_cols + metric_order
    ]
    df = df[ordered_cols]

    # --- Write back to Excel ---
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

def process_all():
    for fname in os.listdir(RESULTS_DIR):
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

    print("✅ All results processed and Excel files updated.")


if __name__ == "__main__":
    process_all()
