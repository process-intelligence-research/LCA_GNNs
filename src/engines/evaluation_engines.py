# For nice result table
import os

import pandas as pd
from sklearn.metrics import r2_score

import wandb


def save_and_evaluate_results(
    results: dict | pd.DataFrame,
    path: str,
    dataset_type: str,
    suffix: str,
    experiment_prefix: str = "",
    task: str | None = None,
) -> None:
    """
    Save results to Excel and run evaluation.

    Parameters
    ----------
    results : Union[Dict, pd.DataFrame]
        Results from training.
    path : str
        Base path to the project directory.
    dataset_type : str
        Type of dataset ("QSPR", "GNN_M", "GNN_C", "GNN_E").
    suffix : str
        Suffix for the output filename.
    experiment_prefix : str, optional
        Prefix for experiment names.
    task : str, optional
        Task name for single-task evaluation.
    """
    # Convert dict results to DataFrame if needed
    if isinstance(results, dict):
        results = pd.DataFrame.from_dict(results)

    # Save to Excel in trained_models folder
    output_file = os.path.join(
        path,
        "trained_models",
        f"{experiment_prefix}{dataset_type}_{suffix}.xlsx",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results.to_excel(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # Run evaluation
    mode = "single" if task or dataset_type in ["QSPR", "GNN_M"] else "multi"
    evaluation_unified(results, mode=mode, task=task, model_type=dataset_type.lower())


def evaluation_unified(
    results, mode="single", task=None, model_type="qspr"
) -> pd.DataFrame:
    """
    Unified evaluation function for all model types and tasks.

    Args:
        results: DataFrame with prediction and ground truth columns
        mode: "single" for single-task or "multi" for multi-task learning
        task: specific task name for single-task mode (e.g., "CC", "AC", etc.)
        model_type: "qspr", "gnn_m", "gnn_c", or "gnn_e" for different model types

    Returns
    -------
        DataFrame with evaluation metrics (MAE, MRE, R2)
    """
    # Handle NaN values for single-task models
    if mode == "single":
        results = results.fillna(0)

    # Define categories based on mode and task
    if mode == "single":
        if task is None:
            task = "CC"  # Default for qspr and gnn_m
        category = [task]
    else:  # multi-task
        category = [
            "AC",
            "CC",
            "ECO",
            "ER",
            "EUf",
            "EUm",
            "EUt",
            "HT",
            "IR",
            "LU",
            "MR",
            "OD",
            "PMF",
            "POF",
            "WU",
        ]

    num_categories = len(category)

    # Calculate MAE
    dic = {}
    for i in range(int(len(results.columns.values) / 2)):
        dic[results.columns.values[i * 2]] = abs(
            results[results.columns.values[i * 2]]
            - results[results.columns.values[i * 2 + 1]]
        )
    AE_dic = pd.DataFrame(dic)

    if mode == "single":
        MAE = {
            value: key
            for key, value in zip(
                AE_dic.mean(axis=0)
                .values.reshape((int(len(results.columns.values) / 2), -1))
                .mean(axis=1),
                category,
            )
        }
    else:
        MAE = {
            value: key
            for key, value in zip(
                AE_dic.mean(axis=0).values.reshape((num_categories, -1)).mean(axis=1),
                category,
            )
        }

    # Calculate MRE
    dic = {}
    for i in range(int(len(results.columns.values) / 2)):
        dic[results.columns.values[i * 2]] = (
            abs(
                (
                    results[results.columns.values[i * 2]]
                    - results[results.columns.values[i * 2 + 1]]
                )
                / results[results.columns.values[i * 2 + 1]]
            )
            * 100
        )
    RE_dic = pd.DataFrame(dic)

    if mode == "single":
        MRE = {
            value: key
            for key, value in zip(
                RE_dic.mean(axis=0)
                .values.reshape((int(len(results.columns.values) / 2), -1))
                .mean(axis=1),
                category,
            )
        }
    else:
        MRE = {
            value: key
            for key, value in zip(
                RE_dic.mean(axis=0).values.reshape((num_categories, -1)).mean(axis=1),
                category,
            )
        }

    # Calculate R2
    dic = {}
    for i in range(int(len(results.columns.values) / 2)):
        dic[results.columns.values[i * 2]] = r2_score(
            results[results.columns.values[i * 2 + 1]],
            results[results.columns.values[i * 2]],
        )
    r2_dic = pd.DataFrame(dic, index=[0])

    if mode == "single":
        R2 = {
            value: key
            for key, value in zip(
                r2_dic.mean(axis=0)
                .values.reshape((int(len(results.columns.values) / 2), -1))
                .mean(axis=1),
                category,
            )
        }
    else:
        R2 = {
            value: key
            for key, value in zip(
                r2_dic.mean(axis=0).values.reshape((num_categories, -1)).mean(axis=1),
                category,
            )
        }

    df = pd.DataFrame([MAE, MRE, R2], index=["MAE", "MRE", "R2"]).T
    print(df)

    # Log metrics to wandb
    if mode == "single":
        wandb.log(
            {
                "MAE": MAE,
                "MRE": MRE,
                "R2": R2,
            }
        )
    else:
        wandb.log({"Result_table": wandb.Table(dataframe=df)})

    if mode == "multi":
        return results
    else:
        return df


def evaluation_qspr(results) -> pd.DataFrame:
    """
    Legacy wrapper for QSPR evaluation. Use evaluation_unified instead.
    """
    return evaluation_unified(results, mode="single", task="CC", model_type="qspr")


def evaluation_gnn_m(results) -> pd.DataFrame:
    """
    Legacy wrapper for GNN molecular evaluation. Use evaluation_unified instead.
    """
    return evaluation_unified(results, mode="single", task="CC", model_type="gnn_m")


def evaluation_gnn_c_single(results, task) -> pd.DataFrame:
    """
    Legacy wrapper for GNN country-specific single-task evaluation. Use evaluation_unified instead.
    """
    return evaluation_unified(results, mode="single", task=task, model_type="gnn_c")


def evaluation_gnn_e_single(results, task) -> pd.DataFrame:
    """
    Legacy wrapper for GNN energy-specific single-task evaluation. Use evaluation_unified instead.
    """
    return evaluation_unified(results, mode="single", task=task, model_type="gnn_e")


def evaluation_gnn_c_multi(results) -> pd.DataFrame:
    """
    Legacy wrapper for GNN country-specific multi-task evaluation. Use evaluation_unified instead.
    """
    return evaluation_unified(results, mode="multi", model_type="gnn_c")


def evaluation_gnn_e_multi(results) -> pd.DataFrame:
    """
    Legacy wrapper for GNN energy-specific multi-task evaluation. Use evaluation_unified instead.
    """
    return evaluation_unified(results, mode="multi", model_type="gnn_e")
