from data.dataset_loader import MedicalDatasetLoader, prepare_evaluation_data

loader = MedicalDatasetLoader()

# Charger tous les datasets
datasets = loader.get_all_datasets()

# Préparer données d'évaluation
inputs, ground_truth, metadata = prepare_evaluation_data(datasets)

print(f"\n📊 Evaluation data prepared:")
print(f"   Total samples: {len(inputs)}")
print(f"   Unique diseases: {len(set(ground_truth))}")

# Afficher exemples
print("\n📋 Sample data:")
for i in range(min(3, len(inputs))):
    print(f"\n{i+1}. Input: {inputs[i][:100]}...")
    print(f"   Ground truth: {ground_truth[i]}")
    print(f"   Source: {metadata[i]['source']}")