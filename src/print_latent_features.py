import pandas as pd

path = "data/CP/preop_binary_elasticnet_l1ratio_0.3_feature_importance.csv"
df = pd.read_csv(path)

# Keep latent features only
df_latent = df[df["group"] == "latent"].copy()

# Sort by importance (descending)
df_latent = df_latent.sort_values("importance", ascending=False)

# Take top K latent channels
K = 10
top_latent = df_latent.head(K)

print(top_latent[["feature_name", "importance", "latent_channel_idx"]])
top_channel_indices = top_latent["latent_channel_idx"].to_numpy(dtype=int)
print(top_channel_indices)
