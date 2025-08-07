import json
import os
import json
import os

def visualize_coi_scores(json_path, sort_by="F1 Score", decimal_places=2):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file {json_path}: {e}")
        return
    except Exception as e:
        print(f"Unexpected error reading file {json_path}: {e}")
        return

    if not datas:
        print(f"No data found in file {json_path}.")
        return

    # Get first model key to access metrics
    first_model_key = next(iter(datas))
    metrics_order = list(datas[first_model_key].keys())
    if sort_by not in metrics_order:
        print(f"Warning: Sort metric '{sort_by}' not found in data. Using first metric '{metrics_order[0]}' instead.")
        sort_by = metrics_order[0]

    print(f"--- Scores from {os.path.basename(json_path)} (sorted by {sort_by}) ---")

    header_string = 'Method\t\t& ' + '\t& '.join(metrics_order) + ' \\\\'
    print(header_string)
    print("\\hline")

    try:
        # Sort models by specified metric 
        sorted_models = sorted(
            datas.keys(),
            key=lambda model_name: (
                datas[model_name].get(sort_by, float('-inf'))
                if isinstance(datas[model_name].get(sort_by), (int, float))
                else float('-inf')
            ),
            reverse=True  # Sort in descending order
        )
    except Exception as e:
        # Fallback to unsorted if sorting fails
        sorted_models = list(datas.keys())

    # Calculate max model name length for alignment
    max_model_name_len = 0
    if sorted_models:
        max_model_name_len = max(len(name) for name in sorted_models)
    max_model_name_len = max(max_model_name_len, 4)

    # Print scores for each model
    for model_name in sorted_models:
        model_data = datas[model_name]
        scores_for_row = []
        for metric in metrics_order:
            score_val = model_data.get(metric)
            if isinstance(score_val, (int, float)):
                # Format numeric scores with specified decimal places
                scores_for_row.append(f"{score_val:.{decimal_places}f}")
            else:
                # Handle non-numeric or missing values
                scores_for_row.append(str(score_val) if score_val is not None else "N/A")
        print(f"{model_name:<{max_model_name_len}}\t& " + '\t& '.join(scores_for_row) + ' \\\\')

    print("===================================================")


cor_json_path = 'CogStream/CoR.json'
# Call the visualization function
print("\n--- Visualizing COI Scores (Sorted by F1 Score) ---")
visualize_coi_scores(cor_json_path, sort_by="F1 Score", decimal_places=2)

print("\n--- Visualizing COI Scores (Sorted by Accuracy, 3 decimal places) ---") 
visualize_coi_scores(cor_json_path, sort_by="Accuracy", decimal_places=3)

print("\n--- Visualizing COI Scores (Attempting to sort by a non-existent metric) ---")
visualize_coi_scores(cor_json_path, sort_by="NonExistentMetric")
