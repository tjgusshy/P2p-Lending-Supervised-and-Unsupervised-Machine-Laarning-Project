# Chapter 3: Model Development Methodology

## 3.1 Introduction

This chapter presents a comprehensive methodology for developing and evaluating credit risk classification models for peer-to-peer lending. The approach implements supervised machine learning techniques with rigorous validation protocols to ensure model reliability and generalization capability. The methodology addresses critical challenges in credit risk modelling, including class imbalance, feature heterogeneity, and model interpretability.

## 3.2 Target Variable and Feature Engineering

### 3.2.1 Target Variable Definition

The binary classification target variable `loan_grade_status` was defined to represent loan outcome:
- **Class 0 (Paid)**: Loans that were successfully repaid
- **Class 1 (Default)**: Loans that defaulted or were charged off

The target variable was extracted from the prepared dataset `df_model`, which underwent preprocessing and feature engineering as described in Chapter 2.

### 3.2.2 Feature Type Identification

Features were systematically categorized into two groups based on their data types:

**Numeric Features**: Continuous and discrete quantitative variables including:
- Financial metrics (annual income, loan amount, interest rate)
- Credit history indicators (delinquency counts, inquiries)
- Debt ratios and payment-to-income metrics
- Employment length (in months)

**Categorical Features**: Nominal variables representing:
- Homeownership status (rent, mortgage, own)
- Income verification status
- Loan purpose categories
- Application type (individual vs joint)
- Geographic region

This dual-type feature structure necessitates distinct preprocessing pipelines, as categorical variables require encoding while numeric variables require standardization (Géron, 2019).

### 3.2.3 Class Distribution Analysis

Prior to model development, the target variable distribution was examined to quantify class imbalance. The analysis revealed a significant imbalance between paid and defaulted loans, which is characteristic of credit risk datasets where defaults are minority events (Hand and Henley, 1997). This imbalance requires specialized handling to prevent model bias toward the majority class.

## 3.3 Data Partitioning Strategy

### 3.3.1 Stratified Train-Test Split

A stratified random sampling approach was employed to partition the dataset:

```
Train Set: 80% of observations
Test Set: 20% of observations
Random Seed: 42 (for reproducibility)
Stratification: By target variable
```

**Rationale for Stratification**: Stratified sampling ensures that both training and test sets maintain the same proportion of defaulted to paid loans as the original dataset (Kohavi, 1995). This is critical for:
1. Preventing sampling bias in the minority class
2. Ensuring representative performance estimates
3. Maintaining statistical power for model evaluation

The test set serves as a held-out validation set that remains unseen during model training and hyperparameter tuning, providing an unbiased estimate of generalization performance.

## 3.4 Data Preprocessing Pipeline

### 3.4.1 Numeric Feature Transformation

Numeric features were standardized using `StandardScaler`, which applies z-score normalization:

```
z = (x - μ) / σ
```

Where:
- `x` = original feature value
- `μ` = mean of the feature (computed on training data only)
- `σ` = standard deviation of the feature

**Justification**: Standardization is essential for:
- Distance-based algorithms (logistic regression uses gradient descent)
- Ensuring features contribute proportionally to model learning
- Improving numerical stability and convergence speed (Bishop, 2006)

### 3.4.2 Categorical Feature Encoding

Categorical variables were encoded using one-hot encoding (`OneHotEncoder`):
- Creates binary dummy variables for each category
- Handles unknown categories in test data gracefully (`handle_unknown='ignore'`)
- Uses sparse matrix representation for memory efficiency

**Alternative Encoding Consideration**: One-hot encoding was preferred over label encoding because the categorical variables in this dataset are nominal (no inherent ordering). For tree-based models, categorical encoding schemes are less critical, but for linear models (logistic regression), one-hot encoding prevents false ordinality assumptions (Potdar et al., 2017).

### 3.4.3 Integrated Preprocessing with ColumnTransformer

The `ColumnTransformer` from scikit-learn was utilized to apply different preprocessing strategies to numeric and categorical features simultaneously. This ensures:
1. Correct transformation order (fit on train, transform on test)
2. Prevention of data leakage
3. Seamless integration with scikit-learn pipelines

## 3.5 Handling Class Imbalance

Class imbalance is a fundamental challenge in credit risk modelling, as defaulted loans constitute a minority of observations. Three distinct strategies were implemented across different algorithms:

### 3.5.1 Class Weighting for Linear and Tree Models

For algorithms that support native class weighting (`LogisticRegression` and `RandomForestClassifier`), the `class_weight='balanced'` parameter was employed. This automatically computes weights inversely proportional to class frequencies:

```
w_j = n_samples / (n_classes × n_samples_j)
```

Where:
- `w_j` = weight for class j
- `n_samples` = total number of samples
- `n_classes` = number of classes (2 for binary)
- `n_samples_j` = number of samples in class j

**Effect**: This penalizes misclassification of the minority class (defaults) more heavily during training, encouraging the model to learn discriminative patterns for rare events (He and Garcia, 2009).

### 3.5.2 Sample Weighting for Gradient Boosting

`GradientBoostingClassifier` does not support a `class_weight` parameter. To address this limitation, a custom wrapper class `GradientBoostingWithSampleWeight` was developed. This wrapper:

1. Computes balanced sample weights using `compute_sample_weight(class_weight='balanced', y=y_train)`
2. Passes these weights to the `fit()` method via the `sample_weight` parameter
3. Maintains compatibility with scikit-learn's API (inherits from `BaseEstimator` and `ClassifierMixin`)

**Technical Implementation**: The wrapper ensures that during training, each observation is weighted according to the inverse frequency of its class. This effectively upweights minority class samples without resampling (Chawla et al., 2002).

### 3.5.3 Scale Positive Weight for XGBoost

XGBoost uses a different mechanism for handling imbalance: the `scale_pos_weight` parameter. This was computed as:

```
scale_pos_weight = count(negative_class) / count(positive_class)
```

**Interpretation**: This ratio (typically > 1 for imbalanced data) increases the penalty for misclassifying positive (default) cases. For example, if defaults represent 20% of loans, `scale_pos_weight = 4`, meaning each default carries 4 times the weight of a paid loan during gradient computation (Chen and Guestrin, 2016).

### 3.5.4 Justification for Approach

The methodology deliberately **excludes synthetic oversampling techniques** (e.g., SMOTE) for the following reasons:
1. **Risk of overfitting**: Synthetic samples can introduce artificial patterns
2. **Computational efficiency**: Reweighting avoids dataset expansion
3. **Interpretability**: Original observations are preserved
4. **Regulatory compliance**: Real data is preferable for auditable models in finance (Baesens et al., 2003)

## 3.6 Model Selection and Configuration

### 3.6.1 Candidate Algorithm Portfolio

Four algorithms representing diverse learning paradigms were selected:

#### 3.6.1.1 Logistic Regression
- **Type**: Generalized linear model
- **Interpretability**: High (coefficients directly interpretable as log-odds)
- **Configuration**:
  - L2 regularization (ridge)
  - Maximum iterations: 1000
  - Class weighting: balanced
  - Solver: lbfgs (efficient for L2)

**Rationale**: Logistic regression serves as a strong baseline and is widely used in credit scoring due to its transparency and regulatory acceptability (Thomas et al., 2017).

#### 3.6.1.2 Random Forest
- **Type**: Ensemble of decision trees (bagging)
- **Interpretability**: Medium (feature importance via impurity decrease)
- **Configuration**:
  - Maximum depth: 10 (prevents overfitting)
  - Minimum samples per leaf: 10 (avoids pure leaves)
  - Class weighting: balanced
  - Parallel processing: enabled (n_jobs=-1)

**Rationale**: Random forests are robust to non-linear relationships and feature interactions, making them suitable for complex credit risk patterns (Breiman, 2001).

#### 3.6.1.3 Gradient Boosting (with Sample Weighting)
- **Type**: Ensemble of decision trees (boosting)
- **Interpretability**: Medium-low (sequential additive model)
- **Configuration**:
  - Subsample: 0.8 (stochastic gradient boosting)
  - Early stopping: 10 iterations without improvement
  - Validation fraction: 0.1 (for early stopping)
  - Custom sample weighting wrapper

**Rationale**: Gradient boosting often achieves superior predictive performance by iteratively correcting errors of weak learners (Friedman, 2001).

#### 3.6.1.4 XGBoost
- **Type**: Optimized gradient boosting implementation
- **Interpretability**: Medium-low (tree ensemble with regularization)
- **Configuration**:
  - Learning rate: 0.1
  - Maximum depth: 4 (shallow trees prevent overfitting)
  - Subsample: 0.8
  - Column subsample: 0.8
  - Scale positive weight: computed from class ratio
  - Evaluation metric: log loss

**Rationale**: XGBoost incorporates advanced regularization and parallel processing, often outperforming standard gradient boosting in financial applications (Chen and Guestrin, 2016; Lessmann et al., 2015).

### 3.6.2 Pipeline Architecture

Each algorithm was embedded in a scikit-learn `Pipeline` with two sequential stages:
1. **Preprocessor**: ColumnTransformer (numeric standardization + categorical encoding)
2. **Model**: Configured classifier

**Advantages of Pipeline Design**:
- Prevents data leakage (transformations fit only on training folds during CV)
- Ensures reproducibility
- Simplifies hyperparameter tuning (entire pipeline can be searched)
- Facilitates deployment (single object encapsulates full workflow)

## 3.7 Model Evaluation Framework

### 3.7.1 Cross-Validation Strategy

**5-Fold Stratified Cross-Validation** was employed on the training set to obtain robust performance estimates:

- **Number of folds**: 5 (standard practice balancing bias-variance tradeoff)
- **Stratification**: Each fold maintains original class proportions
- **Shuffling**: Enabled with fixed random seed (42)

**Procedure**:
1. Training set divided into 5 stratified folds
2. For each fold:
   - 4 folds used for training
   - 1 fold held out for validation
3. Performance metrics averaged across all 5 folds
4. Standard deviations computed to assess stability

**Justification**: Cross-validation provides a more reliable estimate of generalization performance than a single train-validation split, especially for imbalanced datasets (Kohavi, 1995).

### 3.7.2 Performance Metrics

Two primary metrics were selected based on their relevance to imbalanced binary classification in credit risk:

#### 3.7.2.1 Area Under the ROC Curve (AUC-ROC)

The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Sensitivity) against False Positive Rate (1 - Specificity) across all classification thresholds. The Area Under the Curve (AUC) summarizes discriminative ability:

- **Range**: 0.5 (random classifier) to 1.0 (perfect classifier)
- **Interpretation**: Probability that a randomly chosen default is ranked higher than a randomly chosen paid loan
- **Threshold-independent**: Evaluates ranking quality, not fixed classification decisions

**Advantages for Credit Risk**:
- Robust to class imbalance (unlike accuracy)
- Reflects model's ability to rank-order risk (critical for portfolio management)
- Standard metric in credit scoring literature (Baesens et al., 2003)

#### 3.7.2.2 Kolmogorov-Smirnov (KS) Statistic

The KS statistic measures the maximum separation between cumulative distribution functions of predicted probabilities for positive and negative classes:

```
KS = max|F_1(x) - F_0(x)|
```

Where:
- `F_1(x)` = CDF of predicted probabilities for defaults
- `F_0(x)` = CDF of predicted probabilities for non-defaults

**Range**: 0 (no discrimination) to 1 (perfect separation)

**Industry Standard**: KS is widely used in credit risk modelling as it directly quantifies class separation. Values above 0.40 are considered strong in financial applications (Siddiqi, 2006).

**Custom Scorer Implementation**: A custom scikit-learn scorer was created using `make_scorer()` to enable KS computation during cross-validation:

```python
from scipy.stats import ks_2samp
ks_scorer = make_scorer(
    lambda y_true, y_pred: ks_2samp(
        y_pred[y_true == 1], 
        y_pred[y_true == 0]
    ).statistic,
    needs_proba=True
)
```

### 3.7.3 Why Not Precision-Recall AUC?

While PR-AUC is valuable for imbalanced datasets, this methodology prioritizes **ROC-AUC and KS** for the following reasons:

1. **Threshold flexibility**: Credit risk models require adjustable decision thresholds based on risk appetite; ROC-AUC evaluates performance across all thresholds
2. **Industry convention**: Financial institutions standardize on AUC and KS for model comparison and regulatory reporting
3. **Interpretability**: KS provides intuitive separation metrics that translate directly to business decisions (e.g., score cutoffs for approval rates)

### 3.7.4 Test Set Evaluation

After cross-validation, models were retrained on the entire training set and evaluated on the held-out test set. For each model, the following were computed:

**Probability-Based Metrics** (primary focus):
- AUC-ROC
- KS Statistic

**Classification Metrics** (at default 0.5 threshold):
- Confusion matrix (TP, TN, FP, FN)
- Precision, Recall, F1-Score (via classification report)

**Note**: Classification metrics are reported for completeness but are secondary to probability-based metrics, as threshold selection depends on business objectives (e.g., target approval rate, loss tolerance).

## 3.8 Generalization Assessment

### 3.8.1 Train-Test Performance Comparison

For each model, training and test set metrics were compared to detect overfitting:

- **Training Metrics**: AUC, KS computed on full training set
- **Test Metrics**: AUC, KS computed on held-out test set
- **Drop-off**: Difference between training and test performance

**Interpretation Criteria**:
- **AUC drop > 0.10**: High risk of overfitting
- **AUC drop 0.03–0.10**: Moderate overfitting (common, but monitor)
- **AUC drop < 0.03**: Good generalization

### 3.8.2 Statistical Validation via Bootstrap

To rigorously test whether observed train-test differences are statistically significant, a **bootstrap hypothesis test** was conducted:

**Procedure**:
1. For each model, extract predicted probabilities on train and test sets
2. Perform 200 bootstrap iterations:
   - Resample train probabilities with replacement → compute AUC
   - Resample test probabilities with replacement → compute AUC
3. Collect bootstrap distributions of train and test AUCs
4. Conduct Welch's t-test (allows unequal variances) on distributions
5. Compute p-value for null hypothesis: "train AUC = test AUC"

**Decision Rule**:
- **p-value ≥ 0.05**: No significant difference → good generalization
- **p-value < 0.05**: Significant difference → evidence of overfitting

**Composite Ranking**: Models were ranked by:
1. Highest p-value (strong generalization evidence)
2. Smallest AUC drop (minimal performance degradation)

This dual-criterion approach identifies models that both perform well and generalize reliably (Efron and Tibshirani, 1993).

## 3.9 Hyperparameter Optimization

### 3.9.1 Focus on Logistic Regression

Following baseline evaluation, **Logistic Regression** was selected for hyperparameter tuning due to:
1. **Interpretability**: Transparent coefficients for regulatory compliance
2. **Efficiency**: Fast training enables extensive grid search
3. **Stability**: Less prone to overfitting than complex ensembles
4. **Baseline strength**: Competitive initial performance

### 3.9.2 Grid Search Strategy

A comprehensive grid search was conducted using `GridSearchCV`:

**Search Space**:
- **Regularization type** (`penalty`):
  - `'l2'` (Ridge): Shrinks coefficients toward zero (suitable for correlated features)
  - `'l1'` (Lasso): Induces sparsity by zeroing weak coefficients (feature selection)
  
- **Regularization strength** (`C`):
  - Values: `[0.01, 0.1, 1, 10]`
  - **Interpretation**: `C = 1/λ`, where λ is the regularization parameter
  - Smaller C → stronger regularization (more shrinkage)
  - Larger C → weaker regularization (closer to unregularized model)
  
- **Solver**:
  - `'lbfgs'` for L2 (efficient quasi-Newton method)
  - `'liblinear'` for L1 (coordinate descent, required for L1)
  
- **Class weighting**:
  - `None` vs. `'balanced'` (tests impact of imbalance handling)

**Total Combinations**: 2 penalties × 4 C values × 2 class weights = **16 configurations**

### 3.9.3 Cross-Validation Configuration

- **Folds**: 5-fold stratified CV (consistent with baseline evaluation)
- **Scoring**: `'roc_auc'` (primary optimization metric)
- **Parallel processing**: Enabled (`n_jobs=-1`)
- **Verbosity**: Level 2 (detailed progress reporting)

**Selection Criterion**: Configuration with highest mean CV AUC was selected as the best model.

### 3.9.4 Final Model Evaluation

The best hyperparameter configuration was:
1. Retrained on the full training set
2. Evaluated on both training and test sets using:
   - `evaluate_model_performance()` function (AUC, KS, confusion matrix)
3. Assessed for generalization via train-test AUC comparison
4. Visualized via ROC curve on test set

## 3.10 Visualization and Reporting

### 3.10.1 Confusion Matrices

Confusion matrices were generated for each model (using 0.5 threshold) to visualize:
- **True Positives (TP)**: Correctly identified defaults
- **True Negatives (TN)**: Correctly identified paid loans
- **False Positives (FP)**: Paid loans incorrectly flagged as defaults (Type I error)
- **False Negatives (FN)**: Defaults incorrectly classified as paid (Type II error, more costly)

Heatmaps with green color scales were used to emphasize correct predictions.

### 3.10.2 ROC Curves

Two ROC visualizations were produced:

1. **Comparative ROC Plot**: All four baseline models overlaid on a single plot
   - Enables direct visual comparison of discriminative ability
   - Diagonal reference line (random classifier) included
   - AUC values displayed in legend

2. **Best Model ROC Plot**: Focused plot for the optimized logistic regression model
   - Highlights final model's performance
   - Used for thesis/report inclusion

### 3.10.3 Performance Summary Tables

Structured tables were generated for:
- Cross-validation results (mean AUC and KS per model)
- Test set performance (AUC, KS, precision, recall, F1 per model)
- Generalization assessment (bootstrap p-values, AUC drops)
- Best hyperparameters (for tuned logistic regression)

## 3.11 Methodological Considerations and Limitations

### 3.11.1 Strengths

1. **Rigorous validation**: Stratified CV + held-out test set prevents overoptimistic estimates
2. **Imbalance handling**: Multiple strategies tailored to each algorithm
3. **Statistical rigor**: Bootstrap testing provides formal generalization assessment
4. **Reproducibility**: Fixed random seeds and pipeline architecture ensure repeatability
5. **Industry alignment**: Metrics (AUC, KS) match financial sector standards

### 3.11.2 Limitations

1. **Temporal validity**: Models trained on historical data may not capture future regime shifts (e.g., economic crises)
2. **Threshold selection**: Fixed 0.5 threshold for classification metrics is arbitrary; business-specific thresholds should be determined via cost-benefit analysis
3. **Feature stability**: Model assumes feature distributions remain stable in production
4. **Interpretability trade-off**: Ensemble models (RF, XGBoost) sacrifice some transparency for performance

### 3.11.3 Future Enhancements

- **Calibration**: Apply Platt scaling or isotonic regression to ensure predicted probabilities are well-calibrated
- **Threshold optimization**: Use profit curves or cost-sensitive learning to determine optimal classification thresholds
- **Temporal validation**: Implement time-based splits (train on older data, test on recent data) to assess temporal stability
- **Fairness analysis**: Evaluate model for disparate impact across protected demographic groups (if legally permissible)

## 3.12 Summary

This chapter presented a comprehensive, statistically rigorous methodology for developing credit risk classification models. The approach addresses key challenges in imbalanced binary classification through:

1. **Stratified data partitioning** to ensure representative train-test splits
2. **Tailored preprocessing pipelines** for heterogeneous feature types
3. **Algorithm-specific imbalance handling** (class weights, sample weights, scale_pos_weight)
4. **Robust evaluation** via stratified cross-validation and held-out testing
5. **Formal generalization testing** through bootstrap hypothesis tests
6. **Systematic hyperparameter optimization** for model refinement

The resulting models were evaluated using industry-standard metrics (AUC-ROC, KS Statistic) and assessed for both predictive performance and generalization capability. The methodology provides a transparent, reproducible framework suitable for academic research and practical deployment in credit risk assessment systems.

---

## References

Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J. and Vanthienen, J. (2003) 'Benchmarking state-of-the-art classification algorithms for credit scoring', *Journal of the Operational Research Society*, 54(6), pp. 627–635. doi:10.1057/palgrave.jors.2601545.

Bishop, C. M. (2006) *Pattern Recognition and Machine Learning*. New York: Springer.

Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32. doi:10.1023/A:1010933404324.

Chawla, N. V., Bowyer, K. W., Hall, L. O. and Kegelmeyer, W. P. (2002) 'SMOTE: Synthetic minority over-sampling technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357. doi:10.1613/jair.953.

Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system', in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. San Francisco, CA: ACM, pp. 785–794. doi:10.1145/2939672.2939785.

Efron, B. and Tibshirani, R. J. (1993) *An Introduction to the Bootstrap*. New York: Chapman and Hall/CRC.

Friedman, J. H. (2001) 'Greedy function approximation: A gradient boosting machine', *Annals of Statistics*, 29(5), pp. 1189–1232. doi:10.1214/aos/1013203451.

Géron, A. (2019) *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 2nd edn. Sebastopol, CA: O'Reilly Media.

Hand, D. J. and Henley, W. E. (1997) 'Statistical classification methods in consumer credit scoring: A review', *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, 160(3), pp. 523–541. doi:10.1111/j.1467-985X.1997.00078.x.

He, H. and Garcia, E. A. (2009) 'Learning from imbalanced data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284. doi:10.1109/TKDE.2008.239.

Kohavi, R. (1995) 'A study of cross-validation and bootstrap for accuracy estimation and model selection', in *Proceedings of the 14th International Joint Conference on Artificial Intelligence*. Montreal, Canada: Morgan Kaufmann, pp. 1137–1143.

Lessmann, S., Baesens, B., Seow, H.-V. and Thomas, L. C. (2015) 'Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research', *European Journal of Operational Research*, 247(1), pp. 124–136. doi:10.1016/j.ejor.2015.05.030.

Potdar, K., Pardawala, T. S. and Pai, C. D. (2017) 'A comparative study of categorical variable encoding techniques for neural network classifiers', *International Journal of Computer Applications*, 175(4), pp. 7–9. doi:10.5120/ijca2017915495.

Siddiqi, N. (2006) *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Hoboken, NJ: John Wiley & Sons.

Thomas, L. C., Edelman, D. B. and Crook, J. N. (2017) *Credit Scoring and Its Applications*. 2nd edn. Philadelphia, PA: Society for Industrial and Applied Mathematics.