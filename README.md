# DS 4320 Project 1 - Credit Card Fraud
### Executive Summary
This repository contains the full deliverable for DS 4320 Project 1. The project focuses on detecting fraudulent credit card transactions using a synthetic relational dataset based on the Sparkov fraud simulation schema. It includes four related tables: customers, cards, merchants, and transactions. These tables were generated using parameters informed by Federal Reserve payment fraud research and other published sources. A Random Forest classifier was trained on a joined dataset that includes customer demographics, merchant risk ratings, card utilization ratios, and geographic distance between the cardholder and merchant. Since fraud detection is a highly imbalanced classification problem, model performance was evaluated using PR AUC and ROC AUC rather than accuracy alone. The full pipeline was built in Python, with DuckDB used for relational querying and data processing.
<br>

<br>

---

### Name - Emujin Batzorig
### NetID - kfm8nx
### DOI - [10.5281/zenodo.19211549](https://doi.org/10.5281/zenodo.19211549)
### Press Release
[**New Data Science Approach Could Help Stop Fraudulent Charges Before They Reach Cardholders**](https://github.com/EmujinBat/credit_card_fraud/blob/5c2db0f33277153942cb97daf207c980a99b5503/press_release.md)
### Data - [link to data](https://myuva-my.sharepoint.com/:f:/g/personal/kfm8nx_virginia_edu/IgCs0UB5YbTrSKrlMvDMkRtQAUZt8q0wntciGDvchRpOP9I?e=ojxvF4)
### Pipeline - [Analysis script](https://github.com/EmujinBat/credit_card_fraud/blob/eaf6b096410fe70a6113f0bcb9e69c8587df7dba/pipeline.ipynb)
### License - [MIT](https://github.com/EmujinBat/credit_card_fraud/blob/bae6ecc7f47604c78f124ce1b221162014e1c3e9/LICENSE)
---
| Spec | Value |
|---|---|
| Name | Emujin Batzorig |
| NetID | kfm8nx |
| DOI | [10.5281/zenodo.19211549](https://doi.org/10.5281/zenodo.19211549)
| Press Release | [New Data Science Approach Could Help Stop Fraudulent Charges Before They Reach Cardholders](https://github.com/EmujinBat/credit_card_fraud/blob/5c2db0f33277153942cb97daf207c980a99b5503/press_release.md) |
| Data | [link to data](https://myuva-my.sharepoint.com/:f:/g/personal/kfm8nx_virginia_edu/IgCs0UB5YbTrSKrlMvDMkRtQAUZt8q0wntciGDvchRpOP9I?e=ojxvF4) |
| Pipeline | [analysis code (.ipynb)](https://github.com/EmujinBat/credit_card_fraud/blob/641efda5faa257a1934fc74a188934c9b3c467e7/pipeline.ipynb) [analysis code (.md)](https://github.com/EmujinBat/credit_card_fraud/blob/641efda5faa257a1934fc74a188934c9b3c467e7/pipeline.md)|
| License | [MIT](https://github.com/EmujinBat/credit_card_fraud/blob/bae6ecc7f47604c78f124ce1b221162014e1c3e9/LICENSE) |

---
<br>

<br>

## Problem Definition
### General and Specific Problem
* **General Problem:** Detect credit card fraud in financial transaction data. 
* **Specific Problem:** Build a relational dataset with stakeholders, cards, merchants, and transactions, then use it to predict whether a transaction is fraudulent. 
### Rationale
The original problem is too broad because credit card fraud involves many systems, including payment networks, bank policies, and real time monitoring tools. This project narrows the problem to a specific data science task: predicting whether a transaction is fraudulent based on transaction and entity level features. The problem was also refined from a single table setup to a relational one because fraud is not usually analyzed in isolation. In practice, transactions are interpreted in the context of the customer, the card, and the merchant. Using multiple related tables also makes the project more realistic and allows for stronger feature engineering.
### Motivation
Credit card fraud remains a significant financial problem, especially as digital and card based payments continue to grow. Because of the scale of modern transaction systems, fraud detection cannot rely on manual review alone. Data driven methods are necessary to identify suspicious activity efficiently and early. This project is motivated by the need to better understand which features are useful for fraud detection and how relational data can improve that process. By modeling the relationships between customers, cards, merchants, and transactions, the project reflects how financial data is structured in real settings and shows how that structure can support better fraud prediction.

### Press Release Headline and Link
[**Identifying Suspicious Patterns in Credit Card Transactions**](https://github.com/EmujinBat/credit_card_fraud/blob/1647e91ed219c6527572c7235fbeb9ee26f92d89/press_release.md)

## Domain Exposition


### References

[^1]: Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0118432
[^2]: Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly Media.
[^3]: Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
[^4]: Codd, E. F. (1970). A Relational Model of Data for Large Shared Data Banks. *Communications of the ACM, 13*(6), 377–387. https://doi.org/10.1145/362384.362685
[^5]: Harris, B. (2020). Sparkov Data Generation. Kaggle. https://www.kaggle.com/datasets/kartik2112/fraud-detection
[^6]: Board of Governors of the Federal Reserve System. (2022). *Federal Reserve Payments Study: 2022 Annual Supplement*. https://www.federalreserve.gov/paymentsystems/fr-payments-study.htm
[^7]: Lokanan, M., Tran, V., & Vuong, N. H. (2022). A supervised machine learning algorithm for detecting and predicting fraud in credit card transactions. *Decision Analytics Journal*. https://www.sciencedirect.com/science/article/pii/S2772662223000036
[^8]: Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE Symposium Series on Computational Intelligence*. https://doi.org/10.1109/SSCI.2015.33

### Terminology

| Term | Definition |
|:-----|:-----------|
| Credit Card Fraud | Unauthorized use of a credit card or card information to make transactions without the cardholder's permission. |
| Fraudulent Transaction | A transaction identified as unauthorized or malicious in the dataset. |
| Legitimate Transaction | A normal transaction made by the authorized cardholder. |
| Transaction Amount (`amt`) | The monetary value of a credit card purchase. |
| Transaction Timestamp | The recorded date and time at which a transaction occurred. |
| Fraud Label (`is_fraud`) | Binary variable: 1 = fraud, 0 = legitimate. |
| Class Imbalance | Condition where one class (legitimate) greatly outnumbers the other (fraud), ~98% vs ~2%. |
| PR-AUC | Precision-Recall Area Under Curve — the primary evaluation metric for imbalanced fraud detection.[^1] |
| ROC-AUC | Receiver Operating Characteristic AUC — secondary metric measuring overall discrimination.[^1] |
| False Positive | A legitimate transaction incorrectly flagged as fraudulent. |
| False Negative | A fraudulent transaction incorrectly classified as legitimate. |
| Utilisation Ratio | Transaction amount divided by credit limit — a behavioral signal for fraud. |
| Geographic Distance | Euclidean distance between cardholder and merchant location — a spatial fraud signal. |
| Merchant Risk Rating | A categorical risk tier (low/medium/high) assigned to merchants based on fraud exposure. |
| Feature Engineering | The process of transforming raw relational data into model-ready predictors.[^2] |
| Random Forest | An ensemble tree-based classifier used for fraud detection in this project.[^3] |
| Relational Model | A data structure organized into tables linked by primary and foreign keys.[^4] |
| Sparkov Schema | A widely used synthetic credit card fraud data schema with named, interpretable fields.[^5] |

### Domain Paragraph
This project is in the financial technology domain, where data science is widely used to detect credit card fraud. Banks and payment companies process large numbers of transactions every day and depend on automated models to identify suspicious activity quickly. In supervised fraud detection, models are trained on labeled historical data to learn patterns that separate fraudulent transactions from legitimate ones. One of the main challenges in this domain is class imbalance, since fraudulent transactions make up only a small share of total activity. Because of that, accuracy alone is not a reliable measure of model quality. Fraud detection also depends on relational context. A transaction may be more suspicious depending on the merchant category, the cardholder’s usual spending behavior, and the distance between the merchant and the cardholder’s home location. This project reflects that setting by using a four table relational dataset and a joined feature matrix for prediction.

### Background Reading
| Title | Brief Description | Link |
|-------|------------------|-------|
| Credit Card Fraud Detection (Capital One) | Explains how financial institutions detect and manage credit card fraud using automated monitoring systems that analyze spending behavior and flag unusual transactions. | [Link](https://drive.google.com/file/d/1Bey8jZrmByYqvIJ_12aTj5VuaHbuhfXJ/view?usp=sharing) |
| Credit Card Fraud Detection (SEON) | Describes how fraud detection systems use risk scoring, digital footprint analysis, and machine learning to identify suspicious transactions. | [Link](https://drive.google.com/file/d/1Z3NvHeG4_O0BXJk9dkcb6AUsPQYDryoj/view?usp=share_link) |
| How Machine Learning Works for Payment Fraud Detection (Stripe) | Explains how machine learning models analyze large transaction datasets to identify patterns and anomalies indicating fraud. | [Link](https://drive.google.com/file/d/1WKiwMfcC4YdHVWDw3dB_vFM-W5wLcRwN/view?usp=sharing) |
| Machine Learning for Fraud Detection (Ravelin) | Discusses how modern fraud detection uses ML models analyzing hundreds of customer and transaction characteristics. | [Link](https://drive.google.com/file/d/1Soyt8W5gEHzLriIMUIJPzKb3D1EvWMc5/view?usp=share_link) |
| Machine Learning for Fraud Detection (Coursera) | Provides an overview of how ML techniques apply to financial transaction data to identify fraudulent activity. | [Link](https://drive.google.com/file/d/1JdFT7ZqKlTDzPIl7IVvZSyILqlJo6jmg/view?usp=share_link) |

## Data Creation

### Provenance

The dataset used in this project is synthetic, generated using the Sparkov fraud-simulation schema originally developed by Brandon Harris (available on Kaggle as "Credit Card Transactions Fraud Detection Dataset," kartik2112, 2020). Because the original Kaggle dataset is distributed as a single flat CSV, this project re-implements the Sparkov generation logic in Python (`credit_card_creation.py`) and explicitly separates the data into four normalized relational tables: customers, cards, merchants, and transactions — enabling the relational feature engineering that is the core contribution of this project.

Statistical parameters are grounded in the following sources:

- **Fraud rate (~2%):** The Federal Reserve Payments Study (2022) reports that unauthorized card-not-present transactions represent a small but significant fraction of total card payments, consistent with a ~2% fraud rate used here.
- **Transaction amount distributions:** Lokanan et al. (2022) demonstrate using the Sparkov dataset directly that fraudulent transactions have a higher median amount than non-fraudulent ones, with the fraudulent amount distribution skewed right — supporting the use of a log-normal distribution with a higher mean (μ=5.0, σ=1.2) for fraud versus legitimate transactions (μ=3.5, σ=1.0). Note that this pattern is dataset-dependent; other datasets show the opposite (Dal Pozzolo et al., 2015).
- **Merchant category distribution:** The same Sparkov schema defines 14 merchant categories (e.g., grocery_pos, gas_transport, shopping_net) that reflect realistic spending patterns documented in payment industry research.

The resulting dataset contains 1,000 customers, 1,264 cards, 800 merchants, and 10,000,000 transactions with approximately 2% fraud rate. All tables are stored in Apache Parquet format with Snappy compression via `convert_to_parquet.py`, producing a total dataset exceeding 1 GB. Vectorized NumPy batch generation (500,000 rows per batch) is used to produce 10M rows efficiently without exhausting memory. The fixed random seed (42) ensures full reproducibility — running `credit_card_creation.py` will always produce identical data.

### Code Table
| File | Brief Description | Link |
|------|------------------|-------|
| `credit_card_creation.py` | Generates all four tables (10M transactions) using vectorized NumPy batching, with logging and referential integrity checks. Saves to CSV. | [Link](./credit_card_creation.py) |
| `convert_to_parquet.py` | Converts all four CSV tables to Apache Parquet format (Snappy compression) using chunked reading for memory efficiency. Requires `pyarrow`. | [Link](./convert_to_parquet.py) |
| `pipeline.py` | Loads Parquet tables into DuckDB, engineers features via SQL, trains Random Forest, evaluates and visualizes results. | [Link](./pipeline.py) |

### Bias Identification

Several sources of bias may affect this dataset and the models trained on it.

**Class imbalance:** Fraudulent transactions represent approximately 1.88% of all records. This is realistic but means a naive classifier can achieve 98% accuracy by predicting "legitimate" for every transaction — a well-documented pitfall in fraud detection research.

**Geographic skew:** The simulated customers and merchants are drawn from 20 US cities only. Real card fraud has international dimensions that are absent here, which may cause a model trained on this data to underperform on geographically diverse real-world data.

**Synthetic distribution assumptions:** The log-normal parameters for transaction amounts are derived from published statistics but may not perfectly capture the tails of real fraud distributions — extremely high-value fraud events may be underrepresented.

**Label certainty:** In real datasets, fraud labels reflect detected and confirmed fraud, not every fraudulent transaction. This dataset assigns fraud labels probabilistically based on a fixed rate, which does not capture cases where fraud goes undetected.

### Bias Mitigation

Class imbalance is mitigated by using `class_weight='balanced'` in the Random Forest classifier, which adjusts sample weights inversely proportional to class frequency. Model evaluation relies on PR-AUC and F1-score for the fraud class rather than overall accuracy, which would be misleading. Stratified train-test splitting ensures the fraud rate is preserved in both sets.

Geographic and distributional biases are acknowledged in the provenance documentation. Conclusions drawn from this model should be treated as directional rather than deployment-ready. Future work could incorporate more diverse geographic distributions or import real transaction data to supplement synthetic records.

### Rationale for Critical Decisions

The Sparkov schema was chosen because it includes named and interpretable fields that can be separated into four related tables. Most public fraud datasets are released as single flat files, which makes that kind of relational structure much harder to build. The fraud rate of about 2% was based on the Federal Reserve Payments Study rather than using an artificial 50/50 class split, since preserving class imbalance is important in fraud detection. Transaction amounts were generated with log normal distributions informed by Lokanan et al. (2022), which found that fraudulent transactions in the Sparkov style data tend to be larger on average, though that pattern may not hold in every dataset. The dataset was scaled to 10 million transactions so the dataset is big enough for working with complex ML model, and a fixed random seed of 42 was used to make the full data generation process reproducible.

## Metadata

### Schema — ER Diagram (Logical Level)

```
customers                    cards                        transactions
─────────────────────        ─────────────────────────    ──────────────────────────
PK  customer_id  INT    ──<- PK  card_id       VARCHAR    PK  trans_id    INT
    first        VARCHAR  │      customer_id   INT   FK─<-    trans_num   VARCHAR
    last         VARCHAR  │      cc_num        VARCHAR         card_id     VARCHAR FK──┐
    gender       VARCHAR  │      card_type     VARCHAR         merchant_id VARCHAR FK──│──┐
    dob          DATE     │      credit_limit  FLOAT           category    VARCHAR     │  │
    street       VARCHAR  │      issue_date    DATE            amt         FLOAT       │  │
    city         VARCHAR  │      expiry_date   DATE            trans_date  TIMESTAMP   │  │
    state        VARCHAR  └──────────────────────────         unix_time   INT         │  │
    zip          VARCHAR                                       is_fraud    INT         │  │
    job          VARCHAR                                 ──────────────────────────────┘  │
    city_pop     INT                                                                       │
    lat          FLOAT         merchants                                                   │
    long         FLOAT         ─────────────────────────                                  │
                          ┌─── PK  merchant_id  VARCHAR ->────────────────────────────────┘
                          │        merchant     VARCHAR
                          │        category     VARCHAR
                          │        merch_lat    FLOAT
                          │        merch_long   FLOAT
                          │        merch_city   VARCHAR
                          │        merch_state  VARCHAR
                          │        risk_rating  VARCHAR
                          └─────────────────────────────
```

**Relationships:**
- `customers` 1 → N `cards` (one customer can hold multiple cards)
- `cards` 1 → N `transactions` (one card can have many transactions)
- `merchants` 1 → N `transactions` (one merchant can appear in many transactions)

### Data Table

| Table | Brief Description | Link to Parquet |
|-------|------------------|-------------|
| customers | 1,000 synthetic cardholders with demographics, location, and employment. | [customers.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/kfm8nx_virginia_edu/IQAPIHqPp9fGRr0CuQCppRMcAWth3JPv1GstC4D9lsHxel0?e=1RhdJO) |
| cards | 1,264 credit cards linked to customers, with type, limit, and dates. | [cards.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/kfm8nx_virginia_edu/IQCckdFOjTYOR7HQmTy5Rjn3AdNrBweoj3dr3U1r6m4MJhw?e=ISWFsG) |
| merchants | 800 merchants with category, location, and risk rating. | [merchants.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/kfm8nx_virginia_edu/IQCQIwOVaaXvTqsU-qnw9HDuAchTQthtjhXx6h1jx-8DU4E?e=W7ukP6) |
| transactions | 10,000,000 transactions (~2% fraudulent) linking cards and merchants. >1 GB. | [transactions.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/kfm8nx_virginia_edu/IQAb_7zcfaB2Sb4aw3iptVHXAYraMbYqaG5UHTGinU84Xhw?e=Yay829) |

#### customers

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| customer_id | INTEGER | Unique customer identifier (primary key). | 1 |
| first | VARCHAR | Customer first name. | James |
| last | VARCHAR | Customer last name. | White |
| gender | VARCHAR | Customer gender (M/F). | M |
| dob | DATE | Date of birth. | 1967-04-08 |
| street | VARCHAR | Street address. | 9035 Oak Ave |
| city | VARCHAR | City of residence. | Phoenix |
| state | VARCHAR | State abbreviation. | AZ |
| zip | VARCHAR | 5-digit ZIP code. | 87397 |
| job | VARCHAR | Occupation. | Pharmacist |
| city_pop | INTEGER | Population of the customer's city. | 138306 |
| lat | FLOAT | Customer home latitude. | 41.9388 |
| long | FLOAT | Customer home longitude. | -87.4883 |

#### cards

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| card_id | VARCHAR | Unique card identifier (primary key). | CARD00001 |
| customer_id | INTEGER | Foreign key linking to customers. | 1 |
| cc_num | VARCHAR | 16-digit card number (synthetic). | 5211823793126978 |
| card_type | VARCHAR | Card network (Visa/Mastercard/Amex/Discover). | Mastercard |
| credit_limit | FLOAT | Maximum credit limit in USD. | 6300.00 |
| issue_date | DATE | Date card was issued. | 2016-04-21 |
| expiry_date | DATE | Card expiration date. | 2025-05-01 |

#### merchants

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| merchant_id | VARCHAR | Unique merchant identifier (primary key). | MERCH0001 |
| merchant | VARCHAR | Merchant name. | fraud_national_pro |
| category | VARCHAR | Merchant category (e.g., grocery_pos, travel). | grocery_pos |
| merch_lat | FLOAT | Merchant latitude. | 45.5556 |
| merch_long | FLOAT | Merchant longitude. | -82.8732 |
| merch_city | VARCHAR | Merchant city. | Austin |
| merch_state | VARCHAR | Merchant state. | TX |
| risk_rating | VARCHAR | Fraud risk tier: low / medium / high. | medium |

#### transactions

| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| trans_id | INTEGER | Unique transaction identifier (primary key). | 1 |
| trans_num | VARCHAR | 16-character MD5 hash transaction reference. | 0438bf0186292afb |
| card_id | VARCHAR | Foreign key linking to cards. | CARD00741 |
| merchant_id | VARCHAR | Foreign key linking to merchants. | MERCH0187 |
| category | VARCHAR | Merchant category at time of transaction. | misc_net |
| amt | FLOAT | Transaction amount in USD. | 43.32 |
| trans_date_trans_time | TIMESTAMP | Date and time of the transaction. | 2019-07-24 20:20:17 |
| unix_time | INTEGER | Unix timestamp of the transaction. | 1563999617 |
| is_fraud | INTEGER | Fraud label: 1 = fraud, 0 = legitimate. | 0 |

### Uncertainty Quantification

| Feature | Type of Uncertainty | Quantification |
|---------|-------------------|----------------|
| amt | Scale/distributional | Log-normal (μ=3.5, σ=1.0 legit; μ=5.0, σ=1.2 fraud). Tails may underrepresent extreme fraud. |
| is_fraud | Label uncertainty | Assigned probabilistically at p=0.02; does not reflect detection-based labeling of real data. |
| lat / long | Sampling uncertainty | Drawn from uniform range over continental US; not stratified by actual population density. |
| credit_limit | Scale uncertainty | Derived from age-based heuristic; may not reflect actual credit scoring distributions. |
| merch_lat / merch_long | Sampling uncertainty | Same uniform US range; geographic clustering of real merchants not captured. |
| city_pop | Sampling uncertainty | Drawn from uniform distribution over [5000, 3M]; actual city size distributions are right-skewed. |
| geo_distance | Derived/propagated | Euclidean proxy — not geodesic; uncertainty propagates from lat/long sampling. |
| utilisation_ratio | Derived | Ratio of amt to credit_limit; inherits uncertainty from both parent features. |

## Pipeline

*(See [pipeline.py](./pipeline.py) and [pipeline.md](./pipeline.md))*

The pipeline loads all four CSV tables into DuckDB, joins them with SQL to engineer relational features (utilisation ratio, geographic distance, merchant risk, customer age), trains a Random Forest classifier with balanced class weights, and evaluates performance using PR-AUC and ROC-AUC. Results are visualized in publication-quality charts saved to `results/`.

To run the full pipeline:

```bash
python credit_card_creation.py   # generate 10M rows → data/*.csv
python convert_to_parquet.py     # convert to Parquet (requires pyarrow) → data/*.parquet
python pipeline.py               # run ML pipeline
```

---

## License

MIT License — see [LICENSE](./LICENSE)


