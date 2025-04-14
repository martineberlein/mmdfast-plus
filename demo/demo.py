from model.ml import RandomForestModel, DecisionTreeModel, SVMModel


if __name__ == '__main__':
    csv_path = "../data/diabetes.csv"

    model = RandomForestModel()
    model.split_and_prepare_data(csv_path, test_size=0.3)
    model.train()
    model.evaluate()
    model_correct, model_wrong = model.get_correct_wrong()
    print("Random Forest - Correct Predictions (first few rows):")
    print(model_correct.head())
    print("\nRandom Forest - Mis-Predictions (first few rows):")
    print(model_wrong.head())