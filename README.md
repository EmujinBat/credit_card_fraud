# DS 4320 Project 1 - Credit Card Fraud
### Executive Summary
This repository contains the full deliverable for DS 4320 Project 1. The project addresses the problem of detecting fraudulent credit card transactions using a synthetic relational dataset modeled on the Sparkov fraud-simulation schema. Four interrelated tables — customers, cards, merchants, and transactions — are generated with statistically grounded parameters drawn from Federal Reserve payment-fraud research and published literature. A Random Forest classifier is trained on a joined feature matrix that includes customer demographics, merchant risk ratings, card utilization ratios, and geographic distance between cardholder and merchant. Model performance is evaluated using PR-AUC and ROC-AUC to account for class imbalance. The complete pipeline runs end-to-end in Python using DuckDB for relational querying.

<br>

<br>

---

### Name - Emujin Batzorig
### NetID - kfm8nx
### DOI - [https://doi.org/10.1000/182](https://doi.org/10.1000/182) BLANK!!!!! x
### Press Release
[**Data Science Project uses GFM to meet spec**](https://github.com/UVADS/DS-4320/tree/main)
### Data - [link to data](https://myuva-my.sharepoint.com/:f:/g/personal/kfm8nx_virginia_edu/IgCs0UB5YbTrSKrlMvDMkRtQAUZt8q0wntciGDvchRpOP9I?e=ojxvF4)
### Pipeline - [analysis code](https://doi.org/10.1000/182)
### License - [MIT](LICENSE.md)
---
| Spec | Value |
|---|---|
| Name | Emujin Batzorig |
| NetID | kfm8nx |
| DOI | [https://doi.org/10.1000/182](https://doi.org/10.1000/182) |
| Press Release | [Data Science Project uses GFM to meet spec](https://github.com/UVADS/DS-4320/tree/main) |
| Data | [link to data](https://doi.org/10.1000/182) |
| Pipeline | [analysis code](https://doi.org/10.1000/182) |
| License | [MIT](LICENSE.md) |

---
<br>

<br>

## Problem Definition
### General and Specific Problem
* **General Problem:** Detect credit card fraud in financial transaction data. 
* **Specific Problem:** Build a relational dataset with stakeholders, cards, merchants, and transactions, then use it to predict whether a transaction is fraudulent. 
### Rationale
Credit card fraud is a very broad problem and can involve many things, such as real-time monitoring, customer behavior, and banking systems. For this project, the problem is narrowed to a supervised classification task using structured transaction data. This makes the project manageable while still addressing an important part of fraud detection. A relational approach is also more useful than a single flat table because fraud is often better understood by connecting information about the customer, card, merchant, and transaction.
### Motivation
Credit card fraud is an important problem because fraudulent transactions cause financial losses for both customers and financial institutions. As the number of digital transactions continues to grow, it becomes harder to detect suspicious activity manually. This makes data-driven fraud detection an important tool. This project is motivated by the idea that fraud may be easier to identify when transactions are analyzed along with related information about the customer, card, and merchant instead of looking at each transaction by itself.

### Press Release Headline and Link
[**Data Science Project uses GFM to meet spec**](https://github.com/UVADS/DS-4320/tree/main)

## Domain Exposition

### References
* GitHub Docs - Basic writing and formatting syntax [^1]
* GitHub Flavored Markdown Spec [^2]
* Markdown Creator's Blog [^3]


[^1]: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
[^2]: https://github.github.com/gfm/
[^3]: https://daringfireball.net/projects/markdown/

### Terminology
| Term | appearance | code |
|:------|:------------:|---:|
|Superscript | 2<sup>nd</sup>| `<sup>nd</sup>`|
|Subscript | 2<sub>nd</sub>| `<sub>nd</sub>`|
|Inline code| `import numpy as np`| \` \` |
|Table justification | use colons in table header row | `:---` or `:---:` or `---:`|

### Background Summary
> [!TIP]
> Did you know you can make these call outs



### Code Highlighting

#### Formatting plain
```
import numpy as np

x = 137

for fruit in fruits:
    print fruit
```


#### Formatting with python
```python
import numpy as np

x = 137

for fruit in fruits:
    print fruit
```



