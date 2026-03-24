# Identifying Suspicious Patterns in Credit Card Transactions

## Hook

Every second, thousands of credit card transactions are processed across the United States — and hidden among them are fraudulent charges that cost consumers and banks billions of dollars each year. As digital payments continue to replace cash, financial institutions face a growing challenge: how do you find the needle of fraud in a haystack of ten million transactions?

## Problem Statement

Credit card fraud remains one of the most costly and persistent challenges in the financial industry. According to the Federal Reserve Payments Study (2022), unauthorized card transactions account for billions in annual losses, and the problem is growing as contactless and online payments become the norm. What makes fraud detection so difficult is that fraudulent transactions often look almost identical to legitimate ones. A $300 purchase at a gas station could be routine — or it could be a criminal testing a stolen card. Simple rules and single-table transaction logs are no longer sufficient. Financial institutions need systems that understand not just the transaction, but the cardholder behind it, the merchant involved, and whether the patterns make sense together.

## Solution Description

This project builds a fraud detection system that takes a relational approach — analyzing credit card transactions not in isolation, but in the context of four interconnected data tables: the customer, their card, the merchant, and the transaction itself. By combining information about a cardholder's demographics and home location, their card's credit limit, the merchant's risk category, and whether the purchase location makes geographic sense, the system can identify suspicious patterns that a single transaction table would miss. A machine learning model — a Random Forest classifier — is trained on this enriched dataset and evaluated using precision-recall metrics specifically designed for fraud detection, where fraudulent transactions are rare but costly. Across 10 million simulated transactions, the model achieves a 10× improvement over random guessing on the fraud class, demonstrating that relational data structure is the key ingredient for meaningful detection.

## Chart

![Transaction Amount Distribution by Fraud Label](./results/amount_distribution.png)

**Figure 1.** Distribution of transaction amounts for legitimate versus fraudulent transactions across 500,000 sampled records. Fraudulent transactions are systematically shifted toward higher dollar amounts — a pattern the model learns to recognize alongside other behavioral and relational signals such as credit utilisation ratio and the geographic distance between the cardholder's home address and the merchant location. This visualization demonstrates that fraud leaves detectable traces when transaction data is analyzed in its full relational context.

*Data generated using the Sparkov fraud-simulation schema, with statistical parameters grounded in the Federal Reserve Payments Study (2022). Model: Random Forest classifier (200 trees, balanced class weights). Primary metric: PR-AUC = 0.197 vs. random baseline of 0.020.*
