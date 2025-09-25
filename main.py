# main.py

import argparse
from data_preprocessing import DataPreprocessor
from model import ScreenTimeModel
from sklearn.model_selection import train_test_split

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kids Screen Time Multi-Label Health Impact Predictor')
    parser.add_argument('--data', type=str, required=True, help='Path to the input CSV file')
    
    # --- UPDATE choices to include 'catboost' ---
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'ensemble', 'catboost'],
                        help='Type of model to train (default: xgboost)')
    
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save model artifacts')
    return parser.parse_args()

def main():
    """Main function to run the data preprocessing, model training, and evaluation pipeline."""
    args = parse_arguments()

    print("=" * 80)
    print("🚀 Starting the Model Training Pipeline")
    print("=" * 80)

    # 1. Data Preprocessing
    preprocessor = DataPreprocessor(data_path=args.data)
    X, y = preprocessor.preprocess()
    
    # 2. Create a final hold-out test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Data split: Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 3. Model Training and Tuning (CV is handled inside the train method)
    model = ScreenTimeModel(model_type=args.model_type)
    model.train(X_train, y_train, target_names=preprocessor.target_names)

    # 4. Final Model Evaluation on the hold-out test set
    test_results = model.evaluate(X_test, y_test)

    # 5. Save the final model pipeline and preprocessor artifacts
    model.save_model(output_dir=args.save_dir)
    preprocessor.save_artifacts(save_dir=args.save_dir) # Saves binarizer and feature names

    print("\n" + "=" * 80)
    print("✅ Pipeline completed successfully!")
    print(f"Model: {args.model_type.upper()}")
    print(f"Final Test Set Metric (F1 Macro): {test_results['f1_macro']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()