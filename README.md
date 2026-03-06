# PaySim Fraud Risk Console

This project builds an end-to-end fraud detection workflow on the PaySim mobile money transaction dataset:
- Descriptive analytics (EDA)
- Predictive modeling (Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP)
- Explainability with SHAP
- Deployed Streamlit app with interactive prediction

Literature Review

Financial fraud detection is a challenging machine learning problem because fraud is both rare and adaptive. In real transaction systems, fraudulent cases are typically a very small share of all observations, which means that a model can appear highly accurate while still failing to catch meaningful fraud. Research therefore emphasizes metrics beyond raw accuracy, especially precision, recall, F1-score, ROC-AUC, and PR-AUC, with PR-AUC often being particularly useful under heavy class imbalance.

A major obstacle in fraud research is the limited availability of public financial transaction data due to privacy and regulatory constraints. The PaySim dataset was introduced to address that problem by simulating mobile money transactions from an original real-world transactional pattern. Lopez-Rojas et al. describe PaySim as a financial simulator designed to generate data that resembles real mobile money systems while making fraud research more reproducible and accessible. That makes PaySim especially valuable for academic experimentation, benchmarking, and model comparison.

Tree-based ensemble methods are especially relevant in fraud detection because they capture non-linear relationships, interactions among variables, and heterogeneous risk patterns better than many linear baselines. Among these methods, XGBoost has become one of the most influential approaches in applied machine learning. Chen and Guestrin show that XGBoost combines scalable gradient boosting with algorithmic optimizations such as sparsity-aware learning and efficient approximate split finding, which helps explain why it performs strongly across many tabular prediction problems, including fraud detection.

At the same time, simpler models such as logistic regression still remain important as baselines because they provide interpretability and establish whether more complex models truly add predictive value. In fraud settings, comparing interpretable baselines with non-linear methods such as decision trees, random forests, and gradient boosting is good practice because it helps demonstrate whether gains come from capturing more complex structure in the data rather than from overfitting or metric inflation. This comparison is especially important on synthetic financial datasets like PaySim, where feature engineering and class imbalance can heavily influence results.

Model explainability is increasingly important in financial applications because predictions may influence downstream operational or compliance decisions. SHAP (SHapley Additive exPlanations), introduced by Lundberg and Lee, provides a theoretically grounded way to assign each feature a contribution to an individual prediction while also supporting global importance views across the dataset. SHAP is particularly useful in fraud detection because it helps analysts understand whether a transaction was flagged due to transaction amount, balance behavior, transaction type, or other engineered signals.

Recent review literature also reinforces that fraud detection should be approached as more than a pure classification exercise. Effective systems must deal with imbalance handling, cost-sensitive decision-making, interpretability, and operational deployment considerations. In practice, this means that a useful fraud model is not simply the one with the highest score on a benchmark, but the one that balances detection quality with transparency and practical usability in a live decision workflow.

Relevance to This Project

This project follows that literature in three ways. First, it uses PaySim as a privacy-preserving yet realistic benchmark for transaction fraud modeling. Second, it compares multiple supervised learning approaches, including logistic regression, decision tree, random forest, MLP, and XGBoost, to evaluate which method performs best under class imbalance. Third, it complements predictive performance with SHAP-based explainability and an interactive Streamlit dashboard so that the final output is not only accurate, but also interpretable and usable for decision support.

References

Lopez-Rojas, E., Elmir, A., & Axelsson, S. PaySim: A Financial Mobile Money Simulator for Fraud Detection. European Modeling and Simulation Symposium, 2016.

Chen, T., & Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.

Lundberg, S. M., & Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.

Ahmad, H. et al. Class balancing framework for credit card fraud detection based on clustering and similarity-based selection. 2022.

Baisholan, N. et al. A Systematic Review of Machine Learning in Credit Card Fraud Detection. 2025.
