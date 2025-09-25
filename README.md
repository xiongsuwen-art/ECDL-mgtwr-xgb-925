# Reproducibility Code

This repository contains two core scripts used for the analyses in the manuscript:

- `WGTWR.py`: Spatiotemporal weighted regression (MGTWR) analysis  
- `XGBoost+SHAP.py`: XGBoost modeling and SHAP interpretation analysis  

## Usage
1. Clone the repository and run the analyses:
   ```bash
   git clone https://github.com/xiongsuwen-art/ECDL-mgtwr-xgb-925.git
   cd ECDL-mgtwr-xgb-925

   # Run MGTWR analysis
   python WGTWR.py

   # Run XGBoost+SHAP analysis
   python "XGBoost+SHAP.py"

