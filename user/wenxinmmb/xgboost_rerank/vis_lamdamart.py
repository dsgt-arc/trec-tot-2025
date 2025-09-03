import xgboost as xgb
from xgboost import plot_tree, plot_importance
import matplotlib.pyplot as plt
import os
import argparse
import json
import tempfile

parser = argparse.ArgumentParser(description="Train a LambdaMART model.")
parser.add_argument("--dir", type=str, required=True, help="Working directory for outputs")
args = parser.parse_args()
# Path to the LambdaMART model
model_path = f"{args.dir}/lambdamart_model.json"

os.makedirs(f"{args.dir}/vis", exist_ok=True)

version = args.dir[-2:]

# Load the model
model = xgb.Booster()
model.load_model(model_path)

# Load features from JSON
with open(f"{args.dir}/info.json") as f:
    info = json.load(f)
features = info["features"]

# Write fmap to a temporary file
with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fmap_file:
    for idx, name in features.items():
        short_name = name.split('(')[0].strip()
        short_name = ''.join(word.capitalize() for word in short_name.split())
        line = f"{int(idx)-1} {short_name} q\n"
        print(repr(line))  # Debug: see exactly what's written
        fmap_file.write(line)
    fmap_path = fmap_file.name

# Visualize the first tree in the model
plt.figure(figsize=(60, 40))
plot_tree(model, tree_index=0, fmap=fmap_path)  # Change num_trees to visualize other trees
# plt.show()

# Save the tree visualization to a file
plt.savefig(f"{args.dir}/vis/lambdamart_tree.png")

# Plot feature importance
plt.figure(figsize=(30, 20))
plot_importance(model, importance_type='weight', fmap=fmap_path, ylabel=None)  # Options: 'weight', 'gain', 'cover'
plt.title(f"Model {version} Feature Importance (weight)")
plt.tight_layout()
# plt.show()

# Save the feature importance plot
plt.savefig(f"{args.dir}/vis/feature_importance_weight.png")

# Add another plot for feature importance 'gain'
plt.figure(figsize=(30, 20))
plot_importance(model, importance_type='gain', fmap=fmap_path, values_format='{v:.2f}',ylabel=None)
plt.title(f"Model {version} Feature Importance (gain)")
plt.tight_layout()
plt.savefig(f"{args.dir}/vis/feature_importance_gain.png")

# Analyze tree depth distribution
tree_info = model.get_dump()
tree_depths = [tree.count('\n') for tree in tree_info]

# Plot tree depth distribution
plt.figure(figsize=(10, 6))
plt.hist(tree_depths, bins=range(min(tree_depths), max(tree_depths) + 1), edgecolor='black')
plt.title(f"Model {version} Tree Depth Distribution")
plt.xlabel("Tree Depth")
plt.ylabel("Frequency")
plt.grid(axis='y')
# plt.show()

# Save the tree depth distribution plot
plt.savefig(f"{args.dir}/vis/tree_depth_distribution.png")