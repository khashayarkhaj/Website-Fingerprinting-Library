import json
import os
import pandas as pd
import numpy as np

def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def process_directory(base_path):
    """Process all JSON files in the directory structure."""
    # Initialize DataFrame with proper structure
    columns = pd.MultiIndex.from_product([
        ['20% loaded', '30% loaded', '40% loaded', '50% loaded', '60% loaded'],
        ['P', 'R', 'F1']
    ])
    df = pd.DataFrame(index=['CW', 'Tik\_Tok', 'Tik\_Tok\_30'], columns=columns)
    
    # Fill DataFrame with data
    for method in ['CW', 'Tik_Tok', 'Tik_Tok_30']:
        method_key = method.replace('_', '\_')  # Escape underscores for LaTeX
        for i, percent in enumerate(['20', '30', '40', '50', '60']):
            file_path = os.path.join(base_path, method, 'Holmes', f'test_p{percent}.json')
            data = load_json_data(file_path)
            
            if data:
                load_key = f'{percent}% loaded'
                df.loc[method_key, (load_key, 'P')] = data['Precision'] * 100
                df.loc[method_key, (load_key, 'R')] = data['Recall'] * 100
                df.loc[method_key, (load_key, 'F1')] = data['F1-score'] * 100
            else:
                load_key = f'{percent}% loaded'
                df.loc[method_key, (load_key, 'P')] = np.nan
                df.loc[method_key, (load_key, 'R')] = np.nan
                df.loc[method_key, (load_key, 'F1')] = np.nan
    
    return df

def generate_latex_table(df):
    """Generate LaTeX code for the table."""
    latex_code = [
        "\\begin{table}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}",
        "\\hline",
        " & \\multicolumn{3}{c|}{20\\% loaded} & \\multicolumn{3}{c|}{30\\% loaded} & \\multicolumn{3}{c|}{40\\% loaded} & \\multicolumn{3}{c|}{50\\% loaded} & \\multicolumn{3}{c}{60\\% loaded} \\\\",
        "Data & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 & P & R & F1 \\\\"
        "\\hline"
    ]
    
    # Add data rows
    for idx in df.index:
        row_values = []
        for load in ['20% loaded', '30% loaded', '40% loaded', '50% loaded', '60% loaded']:
            for metric in ['P', 'R', 'F1']:
                value = df.loc[idx, (load, metric)]
                row_values.append(f"{value:.2f}" if pd.notnull(value) else "-")
        
        latex_code.append(f"{idx} & " + " & ".join(row_values) + " \\\\")
    
    # Add closing
    latex_code.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\caption{Comparison of different methods across various load percentages.}",
        "\\label{tab:results}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

def main():
    base_path = './logs'  # Adjust this to your actual path
    results_df = process_directory(base_path)
    latex_code = generate_latex_table(results_df)
    
    # Print to console
    print(latex_code)
    
    # Save to file
    with open('results_table.tex', 'w') as f:
        f.write(latex_code)
    print("\nLaTeX code has been saved to 'results_table.tex'")

if __name__ == "__main__":
    main()