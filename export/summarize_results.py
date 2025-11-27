import os
import re

import pandas as pd


def filter_model_names(model_names):
        """
        Filter model names by removing the date from the model name.

        Parameters
        ----------
        model_names : list of str
            List of model names.

        Returns
        -------
        filtered_names : list of str
            List of filtered model names.

        Notes
        -----
        The function uses the regular expression '_\d{2}-\d{2}-\d{2}_\d{2}-\d{2}' to filter out the date from the model name.
        """  # noqa: W605
        pattern = re.compile(r"_\d{2}-\d{2}-\d{2}_\d{2}-\d{2}")
        filtered_names = [re.sub(pattern, "", name) for name in model_names]
        return filtered_names

def extract_summary(df, metrics=["cosine_acc", "mover_acc"]):
    """
    Extract useful metrics from a pandas DataFrame containing evaluation results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing evaluation results.
    metrics : list of str, optional
        List of metrics to extract from the DataFrame. Defaults to ['cosine_acc', 'mover_acc'].

    Returns
    -------
    styled_df : pandas.io.formats.style.Styler
        Styler object containing the extracted metrics.

    Notes
    -----
    The function filters out the model name from the DataFrame, sets the index to 'Model' and 'Date', and then concatenates the DataFrames along 'Model' axis.
    Finally, it highlights the maximum values in the DataFrame using the green color, and formats the values to precision 3.

    """

    for sheet in df.keys():
        df_model_names = df[sheet]["Model"]
        cleaned_df_model_names = filter_model_names(df_model_names)
        df[sheet]["Model"] = cleaned_df_model_names
        df[sheet].set_index(["Model", "Date"], inplace=True)

    accs = [df[sheet][metrics] for sheet in df.keys()]

    column_order = ["belebele", "openbookqa", "veritasqa", "truthfulqa", "xstorycloze"]

    concat_df = pd.concat(accs, keys=df.keys(), axis=1).dropna(how="all")
    concat_df = concat_df.reindex(columns=column_order, level=0)
    styled_df = concat_df.style.highlight_max(axis=0, color="lightgreen").format(precision=3)

    return styled_df


def export_lang_dict_to_excel(lang_df_dict, out_path, max_col_width=50):
    """
    Export a dictionary of DataFrames to an Excel file, with one sheet per language.
    The order of the sheets is 'gl' first, followed by the rest in alphabetical order.
    If the DataFrame has a MultiIndex, use it to set the column widths and apply numeric formatting.
    If the DataFrame does not have a MultiIndex, use the standard column names.
    Parameters:
        lang_df_dict (dict): Dictionary of DataFrames, where the key is the language.
        out_path (str): Path to the output Excel file.
        max_col_width (int): Maximum width of a column in the Excel file (default: 50).
    Returns:
        str: The full path to the output Excel file.
    """
    processed = lang_df_dict.copy()
    ordered = []
    if "gl" in processed:
        ordered.append("gl")
    ordered += [language for language in sorted(processed.keys()) if language != "gl"]

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        num_format = workbook.add_format({"num_format": "0.000"})

        for lang in ordered:
            obj = processed[lang]

            if hasattr(obj, "to_excel"):
                obj.to_excel(writer, sheet_name=lang, index=True, float_format="%.3f", freeze_panes=(1, 2))
                df = obj.data
            else:
                df = obj
                df.to_excel(writer, sheet_name=lang, index=True, float_format="%.3f", freeze_panes=(1, 2))

            worksheet = writer.sheets[lang]

            # Index column width
            index_width = min(max(len(str(x)) for x in df.index) + 2, max_col_width)
            worksheet.set_column(0, 0, index_width)

            # MultiIndex columns:
            if isinstance(df.columns, pd.MultiIndex):
                for i, (superscript, metric) in enumerate(df.columns):
                    base_len = max(len(str(superscript)), len(str(metric))) + 2
                    final_width = min(base_len, max_col_width)

                    worksheet.set_column(i + 1, i + 1, final_width)

                    col_series = df[(superscript, metric)]
                    if pd.api.types.is_numeric_dtype(col_series):
                        worksheet.set_column(i + 1, i + 1, final_width, num_format)

            else:
                for i, col in enumerate(df.columns):
                    base_len = len(str(col)) + 2
                    final_width = min(base_len, max_col_width)
                    worksheet.set_column(i + 1, i + 1, final_width)

                    if pd.api.types.is_numeric_dtype(df[col]):
                        worksheet.set_column(i + 1, i + 1, final_width, num_format)

    print(f"Wrote {out_path} with sheets ordered: {ordered}")


def summarize_results_from_path(output_dir, max_col_width=50):
    """
    Summarize results from a directory, saving a combined Excel file.

    Args:
        output_dir (str): Directory containing Excel files with results.
        max_col_width (int, optional): Maximum column width in the exported Excel file. Defaults to 50.

    Returns:
        None
    """
    langs = []
    summed_dfs = []
    for f in os.listdir(output_dir):
        if not (f.startswith("results_") and f.endswith(".xlsx")):
            continue
        lang = f[8:-5]
        df = pd.read_excel(f"{output_dir}/{f}", sheet_name=None)
        print(f"{f} -> {lang}")
        try:
            summary = extract_summary(df)
        except ValueError:
            print(f"  -> Skipping {lang} due to extraction error")
            continue

        langs.append(lang)
        summed_dfs.append(summary)

    summed_dfs_dict = {langs[i]: summed_dfs[i] for i in range(len(langs))}

    export_lang_dict_to_excel(summed_dfs_dict, f"{output_dir}/combined_results.xlsx", max_col_width=max_col_width)

    print("Summary export completed.")

def summarize_surprisal_from_path(input_dir):
    """
    Summarize surprisal results from a directory, saving a combined Excel file.

    Parameters
    ----------
    input_dir : str
        Directory containing the results_surprisal.xlsx file.

    Returns
    -------
    None
    """
    def export_to_excel(df_list, df_names, out_path, max_col_width=50):
        """
        Export a list of DataFrames to an Excel file with a sheet for each DataFrame.

        Parameters
        ----------
        df_list : list
            List of DataFrames to export.
        df_names : list
            List of names for the sheets in the exported Excel file.
        out_path : str
            Path to the output Excel file.
        max_col_width : int, optional
            Maximum column width in the exported Excel file. Defaults to 50.

        Returns
        -------
        None
        """

        assert len(df_list) == len(df_names), "df_list and df_names must have the same length."

        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            num_format = workbook.add_format({"num_format": "0.000"})

            for dataset, name in zip(df_list, df_names):
                obj = dataset

                if hasattr(obj, "to_excel"):
                    obj.to_excel(writer, sheet_name=name, index=True, float_format="%.3f", freeze_panes=(1, 1))
                    df = obj.data
                else:
                    df = obj
                    df.to_excel(writer, sheet_name=name, index=True, float_format="%.3f", freeze_panes=(1, 1))

                worksheet = writer.sheets[name]

                index_width = min(max(len(str(x)) for x in obj.index) + 5, max_col_width)
                worksheet.set_column(0, 0, index_width)

                if isinstance(obj.columns, pd.MultiIndex):
                    for i, (superscript, metric) in enumerate(obj.columns):
                        base_len = max(len(str(superscript)), len(str(metric))) + 2
                        final_width = min(base_len, max_col_width)

                        worksheet.set_column(i + 1, i + 1, final_width)

                        col_series = df[(superscript, metric)]
                        if pd.api.types.is_numeric_dtype(col_series):
                            worksheet.set_column(i + 1, i + 1, final_width, num_format)

    file_name = os.path.join(input_dir, "results_surprisal.xlsx")
    df = pd.read_excel(file_name, sheet_name=None)
    sheets = [df[sheet] for sheet in df.keys()]

    # TODO: This process could be automated by extracting the unique benchmarks directly and associating which metric is not needed
    # benchmarks = [sh["benchmark"].unique().tolist() for sh in sheets]
    # unique_benchmarks = [item for sublist in benchmarks for item in sublist]
    # unique_benchmarks = list(set(unique_benchmarks))

    dfc = df.copy()

    for sheet in dfc.keys():
        dfc_model_names = dfc[sheet]["model"]
        cleaned_dfc_model_names = filter_model_names(dfc_model_names)
        dfc[sheet]["model"] = cleaned_dfc_model_names
        dfc[sheet].set_index(["model"], inplace=True)

    sheets = [dfc[sheet] for sheet in dfc.keys()]
    sheets_cola = [sh[sh["benchmark"] == "cola"].drop(labels=["benchmark", "mean_last_word", "lang"], axis=1) for sh in sheets]
    sheets_calame = [sh[sh["benchmark"] == "calame"].drop(labels=["benchmark", "difsur", "lang"], axis=1) for sh in sheets]

    export_cola = (
        pd.concat(sheets_cola, axis=1, keys=dfc.keys())
        .dropna(how="all", axis=1)
        .swaplevel(axis=1)
        .style.highlight_max(axis=0, color="green")
        .format(precision=3)
    )

    export_calame = (
        pd.concat(sheets_calame, axis=1, keys=dfc.keys())
        .dropna(how="all", axis=1)
        .swaplevel(axis=1)
        .style.highlight_min(axis=0, color="green")
        .format(precision=3)
    )

    dfs_surprisal = [export_cola, export_calame]
    dfs_surprisal_names = ["cola", "calame"]
    out_path=os.path.join(input_dir, "surprisal_summary.xlsx")

    try:
        export_to_excel(dfs_surprisal, dfs_surprisal_names, out_path)
        print(f"✅ Surprisal summary export completed → {out_path}")
    except Exception as e:
        print(f"❌ Error exporting surprisal summary: {e}")
