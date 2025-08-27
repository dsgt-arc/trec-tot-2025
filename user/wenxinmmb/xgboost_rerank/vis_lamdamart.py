import xgboost as xgb
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description="Train a LambdaMART model.")
parser.add_argument("--dir", type=str, required=True, help="Working directory for outputs")
args = parser.parse_args()
# Path to the LambdaMART model
model_path = f"{args.dir}/lambdamart_model.json"

os.makedirs(f"{args.dir}/vis", exist_ok=True)

# Load the model
model = xgb.Booster()
model.load_model(model_path)

# Visualize the first tree in the model
plt.figure(figsize=(30, 20))
plot_tree(model, tree_index=0)  # Change num_trees to visualize other trees
# plt.show()

# Save the tree visualization to a file
plt.savefig(f"{args.dir}/vis/lambdamart_tree.png")

# Plot feature importance
plt.figure(figsize=(10, 8))
plot_importance(model, importance_type='weight')  # Options: 'weight', 'gain', 'cover'
plt.title("Feature Importance")
# plt.show()

# Save the feature importance plot
plt.savefig(f"{args.dir}/vis/feature_importance.png")

# Analyze tree depth distribution
tree_info = model.get_dump()
tree_depths = [tree.count('\n') for tree in tree_info]

# Plot tree depth distribution
plt.figure(figsize=(10, 6))
plt.hist(tree_depths, bins=range(min(tree_depths), max(tree_depths) + 1), edgecolor='black')
plt.title("Tree Depth Distribution")
plt.xlabel("Tree Depth")
plt.ylabel("Frequency")
plt.grid(axis='y')
# plt.show()

# Save the tree depth distribution plot
plt.savefig(f"{args.dir}/vis/tree_depth_distribution.png")