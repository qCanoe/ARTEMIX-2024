# ARTEMIS - TheWebConf24

The repo is for the paper "ARTEMIS: Detecting Airdrop Hunters in NFT Markets with a Graph Learning System" accepted at The Web Conference 2024 (WWW'24).
 
## Abstract

As Web3 projects leverage airdrops to incentivize participation, airdrop hunters tactically amass wallet addresses to capitalize on token giveaways. This poses challenges to the decentralization goal. Current detection approaches tailored for cryptocurrencies overlook non-fungible tokens (NFTs) nuances. We introduce ARTEMIS, an optimized graph neural network system for identifying airdrop hunters in NFT transactions. ARTEMIS captures NFT airdrop hunters through: (1) a multimodal module extracting visual and textual insights from NFT metadata using Transformer models; (2) a tailored node aggregation function chaining NFT transaction sequences, retaining behavioral insights; (3) engineered features based on market manipulation theories detecting anomalous trading. Evaluated on decentralized exchange Blur's data, ARTEMIS significantly outperforms baselines in pinpointing hunters. This pioneering computational solution for an emergent Web3 phenomenon has broad applicability for blockchain anomaly detection. 

## Repository Description

```
.
├── README.md
├── .gitignore
├── data
├── model 
├── notebook
└── scripts
```

- `data`: contains the data used in the experiments.
- `model`: contains the model files.
- `scripts`: contains the train scripts for the experiments, including the ARTMIS model and the baselines.
- `notebook`: contains the jupyter notebook for ARTEMIS training and evaluation on Google Colab.

**Easy training ARTEMIS in Google Colab:** upload all data files to the `data` folder and correctly configure the file paths in `notebook/artemis_train.ipynb`, then run `artemis_train.ipynb` in Google Colab.