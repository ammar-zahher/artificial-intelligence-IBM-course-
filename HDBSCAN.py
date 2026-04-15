import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import geopandas as gpd
import contextily as ctx
import requests
import zipfile
import io
import os

import os

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("ODCAF_v1.0.csv", encoding="ISO-8859-1")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")

df = df.dropna(subset=["Longitude", "Latitude"])
model = HDBSCAN(min_cluster_size=10)
df["Cluster"] = model.fit_predict(df[["Longitude", "Latitude"]])


output_dir = "./"
os.makedirs(output_dir, exist_ok=True)
zip_file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip"
response = requests.get(zip_file_url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith(".tif"):
            zip_ref.extract(file_name, output_dir)


gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326"
)
gdf = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(15, 10))

non_noise = gdf[gdf["Cluster"] != -1]
noise = gdf[gdf["Cluster"] == -1]

if not noise.empty:
    noise.plot(ax=ax, color="k", markersize=5, ec="r", alpha=0.5, label="Noise")

if not non_noise.empty:
    non_noise.plot(ax=ax, column="Cluster", cmap="tab20", markersize=10, alpha=0.7)


ctx.add_basemap(ax, source="./Canada.tif", zoom=4)


plt.title("HDBSCAN Clustering for Canadian Museums Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.show()
