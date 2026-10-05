# Resume & interview notes (delete before publishing if you prefer)

## Resume entry
**AI-Powered Satellite Crop Monitoring System** | Python, Google Earth Engine, Random Forest, XGBoost, QGIS, Streamlit
- Built an end-to-end geospatial ML pipeline that converts Sentinel-2 imagery into crop-stress maps (healthy / water stress / anomaly) using 7 spectral indices (NDVI, EVI, NDRE, NDMI, etc.).
- Trained Random Forest and XGBoost classifiers evaluated with a spatial block hold-out to avoid pixel-autocorrelation leakage; added NDVI change detection and a field-level yield model.
- Delivered QGIS-ready GeoTIFF outputs and an interactive Streamlit dashboard reporting affected area in hectares.

## Be ready to explain
- Why spatial split instead of random split? (neighbouring pixels are correlated → random split inflates accuracy)
- Why NDMI / NDWI catch water stress while NDVI alone lags? (SWIR responds to leaf water content)
- Limits: demo uses synthetic data; real deployment needs field labels, cloud handling, and multi-date features.
- Do NOT claim CNN/U-Net or real-yield results unless you implement/validate them.
