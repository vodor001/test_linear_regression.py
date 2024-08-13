"""Script to do linear regression for odtp demo

Author: Jan Aarts

Usage: python linear_regresssion.py train.csv test.csv instructions.json
"""
import pandas as pd
import json
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression

def load_data(file_path):
    if Path(file_path).suffix == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError("The provided file is not a CSV file.")

def load_instructions(json_path):
    with open(json_path, 'r') as json_file:
        instructions = json.load(json_file)
    return instructions

def perform_linear_regression(train_df, target):
    if target not in train_df.columns:
        raise ValueError(f"The target '{target}' is not a column in the dataframe.")
    
    X_train = train_df.drop(columns=[target]).values
    y_train = train_df[target].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def save_results_to_csv(results, output_path="results.csv"):
    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)

def main():
    if len(sys.argv) != 4:
        print("Usage: python linear_regression_script.py <train_csv> <test_csv> <instructions_json>")
        sys.exit(1)
    
    train_csv_path = sys.argv[1]
    test_csv_path = sys.argv[2]
    json_path = sys.argv[3]
    
    try:
        train_df = load_data(train_csv_path)
        test_df = load_data(test_csv_path)
        instructions = load_instructions(json_path)
        
        target = instructions.get('target')
        if not target:
            raise ValueError("The JSON file must contain a 'target' key.")
        
        model = perform_linear_regression(train_df, target)
        
        # Predictions
        X_test = test_df.drop(columns=[target]).values
        y_test = test_df[target].values
        y_pred = model.predict(X_test)
        
        # Prepare results
        results = {
            'actual': y_test,
            'predicted': y_pred,
            'coef': [model.coef_.tolist()] * len(y_test),
            'intercept': [model.intercept_] * len(y_test)
        }
        
        # Save results to CSV
        save_results_to_csv(results)
        
        print("Results saved to 'results.csv'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
