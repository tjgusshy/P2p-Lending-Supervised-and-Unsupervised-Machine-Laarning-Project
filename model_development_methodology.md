
Chapter 3: Research Methodology
3.1 Introduction
This study uses a hybrid quantitative methodology, combining supervised and unsupervised machine learning techniques, to create a comprehensive credit risk assessment and borrower segmentation framework for the peer-to-peer (P2P) lending industry. The major goal is to move beyond typical default prediction by developing a dual-layered analytical model. The first layer uses a supervised classification model to calculate an individual borrower's Probability of Default (PD). The second layer uses these PD ratings and other borrower variables in an unsupervised clustering model to find separate, actionable borrower segments. This hybrid method, as shown in Figure 3.1, enables both detailed risk assessment at the individual and will increase profits for lenders.
 

Figure 1: Hybrid Methodological Framework



3.2 Research Design and Philosophical Approach
The study adopts a quantitative research design grounded in the positivist paradigm, which emphasizes objective measurement and empirical validation (Saunders, Lewis, & Thornhill, 2019). Positivism is appropriate for this research because the aim is to identify statistically significant patterns in borrower behaviour using numerical data, rather than interpret subjective experiences.
This framework ensures methodological rigour, reproducibility, and transparency, aligning with academic and professional data science standards.
The following steps was followed:
1.	Business understanding – defining the problem (credit risk prediction and segmentation).
2.	Data understanding – exploring and preparing LendingClub data.
3.	Data preparation – cleaning, feature engineering, and transforming inputs.
4.	Modelling – applying Classication algorithms and Segmentation algorithms.
5.	Evaluation – assessing model accuracy and interpretability.
6.	Deployment – developing a Streamlit-based decision-support dashboard.












3.3.1 Structure and Composition of the Dataset
The dataset used in this study originates from the LendingClub public loan dataset, one of the largest and most widely analysed P2P lending datasets in financial research (Jagtiani & Lemieux, 2019; Ko, Kim, & Lee, 2022). It provides detailed information about borrower characteristics, loan attributes, and repayment outcomes.
For the purpose of this project, the dataset was filtered to include only individual borrowers (approximately 80% of all records) and loan amounts ≤ $40,000. This subset represents typical retail lending scenarios on P2P platforms and focuses on transactions most relevant to small investors.The dataset includes 55 variables covering borrower characteristics, loan attributes, credit history, repayment indicators and behavioural factors. These include:
Borrower characteristics
annual income, homeownership status, employment title, employment length, verification status, and debt-to-income ratio. 
Credit history variables
delinquencies, credit lines, utilisation behaviour, public records, collections, number of satisfactory accounts and past due information.
Loan attributes
loan amount, interest rate, grade, sub grade, term, loan purpose, disbursement method and application type
Repayment performance indicators
loan status, paid principal, paid interest, late fees and outstanding balance. Loan status categories include fully paid, charged off and current
 
Figure 2: Shape of the dataset.


3.3.2 Target Variable Considerations
The loan status variable presented significant class imbalance, with less than 2% defaults and 98% non-defaults. The class imbalance problem is of crucial importance since it is encountered by a large number of domains of great environmental, vital or commercial importance, and was shown, in certain cases, to cause a significant bottleneck in the performance attainable by standard learning methods which assume a balanced distribution of the classes (Japkowicz, 2000, p. 111). To mitigate this, credit grade (A to G) was used, which is also highly imbalanced but better than loan status,  82% for non-defaults and 18% for defaults, as shown in the image below.

 

 
 Figure 4.  Loan grade Imbalance Ratio
 
Figure 3.  Loan status Imbalance Ratio








3.3.4 Data Cleaning and Preprocessing
Handling Missing Values: For  NAS and other null values, they were all replaced with np.nan
 
Figure 5: Function for replacing NAS









Replacing badly written data in numeric columns by using pandas to coerce them into numeric variables, which replaces every non numeric variables 
 Figure 6: Shape of the dataset.

















3.3.5 Feature Engineering
To enhance the predictive power of the models, several new features were engineered:
Job Title Categorization: The `emp_title` feature, containing high cardinality, was condensed into a smaller set.
 
Figure 7: Job category mapping






Geographical Binning: The `state` feature was mapped into broader geographical `region` categories (e.g., 'Northeast', 'West') to capture regional economic trends without overfitting to individual states.
 
Figure 8: State mapping to region
















3.3.6 Outlier and Feature Selection
Outlier Treatment: The Winsorization technique was applied to numerical columns identified as having significant outliers. This approach involves modifying the weights of outliers or replacing the values being tested for outliers with expected values. The weight modification method allows weight modification without discarding or replacing the values of outliers, limiting the influence of the outliers.  (Kwak and Kim, 2017).
 
Figure 17: Winsorization of outliers.









Feature Selection: A multi-step feature selection process was implemented to create a quality and good feature set.
 Leakage Prevention: Features that would not be available at the time of a loan application (e.g., `paid_principal`, `last_pymnt_amnt`) were removed to prevent data leakage.
 
Figure 18: Possible columns that can cause leakage.
Multicollinearity Reduction: A correlation matrix was computed for all numeric features. For any pair of features with a correlation coefficient exceeding a threshold of 80% and they were dropped to improve the predictability of the algorithms used.
 
Figure 19: Identification of highly correlated variables.

 
Figure 20: Correlation heatmap 


 
Figure 20: Removal of highly correlated variables.











3.4 Supervised Learning: Probability of Default (PD) Estimation
The first stage is developing a supervised classification model to assign a PD score to each borrower. The approach followed a strict, a process that reduced overfitting, ensured pipeline robustness, and allowed for systematic model comparison. 

 
Figure 20: Removal of highly correlated variables.
3.4.1 Model Selection and Baseline Establishment
Four standard classification algorithms was used to establish a performance baseline:
Logistic Regression
Logistic regression is used to obtain odds ratio in the presence of more than one explanatory variable. The procedure is quite similar to multiple linear regression, with the exception that the response variable is binomial. The result is the impact of each variable on the odds ratio of the observed event of interest. The main advantage is to avoid confounding effects by analysing the association of all variables together (Sperandei, 2014). 

Random Forest
Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large (Breiman, 2001).
Gradient Boosting
Gradient boosting machines are a family of powerful machine-learning techniques that have shown considerable success in a wide range of practical applications. They are highly customizable to the particular needs of the application, like being learned with respect to different loss functions(Natekin and Knoll, 2013).
XGBoost
XGBoost is a powerful and efficient learning method based on an implementation of gradient‐boosted decision trees. It is typically used for supervised learning tasks, particularly regression and classification problems. At a high level, XGBoost works by combining weak learners, such as decision trees, sequentially, with each new learner effectively correcting errors made by the previous ones. Similar to other supervised learning methods, such as neural networks, XGBoost seeks to minimize a loss function, such as mean squared error for regression problems or crossentropy loss for classification problems. (Wiens et al., 2025).


3.3 Data Partitioning Strategy

3.3.1 Stratified Train-Test Split

A stratified random sampling approach was employed to partition the dataset:
Train Set: 80% of observations
Test Set: 20% of observations
Random Seed: 42 (for reproducibility)
The test set serves as a held-out validation set that remains unseen during model training and hyperparameter tuning, providing an unbiased estimate of generalization performance

All models in this study were implemented using the scikit-learn library, which provides a high-level, consistent interface for modern machine learning algorithms and preprocessing tools. Scikit-learn is designed to make state-of-the-art machine learning accessible and reproducible by integrating preprocessing, model fitting, evaluation, and pipeline management within a unified API (Pedregosa et al., 2011). This ensured that numerical scaling, categorical encoding, cross-validation, and hyperparameter tuning were performed in a structured and reproducible manner.
The models were implemented within a `scikit-learn` Pipeline, which encapsulated preprocessing steps (`StandardScaler` for numeric features and `OneHotEncoder` for categorical features to ensure consistent data transformation. 

3.4.3 ColumnTransformer: A Unified Approach to Preprocessing

The ColumnTransformer from scikit-learn was used to apply several preprocessing algorithms to both numeric and categorical information at the same time. This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space. This is useful for heterogeneous or columnar data, to combine several feature extraction mechanisms or transformations into a single transformer. (scikit-learn, 2025).
This ensures 
1.	The right order (fit on train, transform on test).
2.	Prevention of Data Leakage 
3.	Works well with scikit-learn pipelines.

 












3.5 Handling Class Imbalance
As mentioned before, class imbalance is a big issue I experienced which made me change the target variable from loan status to loan grade.   In many real-world applications, dealing with imbalanced data is a critical concern. While most classification methods focus on two-class data problems, addressing a solution for class-imbalanced scenarios is equally essential (Nunna, Panchumarthi and Parchuri, 2024)
 


Each of the model is built to handle imbalances. For Linear (Logistic regression)  and Tree Models (Random forests), class Weighting was used to handle imbalances in these models. The weighted class or class-weight method approaches in a very different way as compared to the pre-existing sampling methods. Instead of creating new or ignoring the existing samples, we here develop as model which would we fed composite inputs comprising of their actual times the inverseof their occurrence(frequency) 
(PDF) Class Weight technique for Handling Class Imbalance (Prakash and Kumar, 2022).
When the class_weights = ‘balanced’, the model automatically assigns the class weights inversely proportional to their respective frequencies.
Formula of Class Weights
wj=n_samples / (n_classes * n_samplesj)
Here,
•	wj is the weight for each class(j signifies the class)
•	n_samplesis the total number of samples or rows in the dataset
•	n_classesis the total number of unique classes in the target
•	n_samplesjis the total number of rows of the respective class
(Analytics Vidhya, 2020).












For Gradient Boosting, a custom wrapper was created to handle the class imbalance, as gradient boosting does not support class_weight directly.
 

For Xgboost, scale_pos_weight is computed to handle the imbalance in the data and then passed to the input of the Xgboost model.
 





3.3.2 Evaluation and Validation
Metrics: Given the credit scoring context, Area Under the Receiver Operating Characteristic Curve (ROC-AUC) and the Kolmogorov-Smirnov (KS) statistic were chosen as the primary evaluation metrics. The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative. The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold (in practice, at selected intervals), then graphing TPR over FPR. (Google Developers, 2025).
 

The KS Score measures how well the model can differentiate between the positive and negative classes based on the distribution of the predictions. It is widely used in industries such as finance, credit, fraud detection, and marketing, where correctly separating different classes can have a significant impact on business (Gaigher, 2023).
 

Other classification metrics is employed to evaluate the model also.
1.	Confusion matrix
2.	Precision
3.	Recall
4.	F1




3.3.3 Model Tuning and Generalisation Testing
Based on its strong baseline performance and interpretability, Logistic Regression was selected for further optimization and based on the model that generalised well and the PD of default used for segemenattion was chosen from logistic regression. The data was trained on the training data and the test data was used to test for genralisation.
Hyperparameter Tuning:`GridSearchCV` was used to systematically search for the optimal combination of hyperparameters ( C, penalty, solver).
Generalisation Assessment: A rule based statistical test was conducted to compare the model's performance on the training set versus the test set. The resulting p-value indicates whether the observed drop in performance is statistically significant.  
 




3.4 Unsupervised Learning: Borrower Segmentation
Segmenting borrowers based on their predicted PD scores and other application-time criteria was the focus of the methodology's second stage.

3.4.1 Hybrid Feature Set Construction
A novel feature set was constructed for the clustering algorithm. It combined:
1.  Borrower Characteristics: A curated set of numeric and categorical features available at the time of application.
2.  Predicted PD Score: The output from the final tuned Logistic Regression model.

3.4.2 Dimensionality Reduction and Clustering
PCA Application: To manage the high-dimensional feature space, Principal Component Analysis (PCA) was applied to the borrower characteristic features (excluding the PD score). Sparse PCA seeks sparse factors, or linear combinations of the data variables, explaining a maximum amount of variance in the data while having only a limited number of nonzero coefficients (Luss and d’Aspremont, 2010).  The PD score was intentionally excluded from PCA to preserve its direct, unadulterated influence on the clustering process.
 
 
-Final Feature Matrix: The resulting principal components were then combined with the standardized PD score to form the final feature matrix for clustering.
 

- Clustering Algorithm: The K-Means algorithm was chosen for its efficiency and interpretability in partitioning the data into a pre-determined number of clusters (k). The optimal value for 'k' was determined using the Elbow Method, which identifies the point of diminishing returns in the Within-Cluster Sum of Squares (WCSS).
 

Figure 3.4: Elbow Method for Optimal k Selection

3.4.3 Segment Profiling and Interpretation
Following their formation, the clusters were profiled to determine their business significance. Each section was given an intuitive risk label ('Low Risk', 'Moderate Risk', and 'High Risk') based on the average PD score within each cluster. The unsupervised output is converted into useful business intelligence in this last stage and saved as pdf file
 

3.5 Decision Support Framework and Implementation
The clusters were profiled once they were formed in order to assess their business significance. Based on the average PD score within each cluster, each area was assigned an intuitive risk label ('Low Risk', 'Moderate Risk', and 'High Risk'). In this final step, the unsupervised output is transformed into valuable business intelligence. A python script was created for the streamlit application.
The DSS is designed to be intuitive, providing users with multiple lenses through which to view the P2P lending landscape:
Interactive Filtering: Users can dynamically filter the loan portfolio based on key criteria such as `risk_level`, `loan_amount`, and `interest_rate`. This allows for the immediate selection of loans that align with a specific investment strategy or risk appetite.
-    Investment Overview
: A central feature of the dashboard is a scatter plot that visualizes the relationship between the predicted Probability of Default (`pd_score`) and the `interest_rate`. This allows investors to visually assess the risk-return trade-off for individual loans and entire segments.
Risk Distribution
: The dashboard provides aggregated metrics for each risk segment, including average interest rates, default probabilities, and a calculated `expected_return`. This high-level summary supports strategic allocation of capital across different risk tiers.
-   Investment Recommendations
: Users can drill down into a detailed, sortable table of individual loans that match their selected criteria, facilitating the final selection process.
Investment Decision Guide

By operationalizing the model outputs in this manner, the research moves beyond theoretical analysis to provide a tangible framework that directly supports the decision-making process in P2P lending.

 Figure 3.6: Investor Decision Support Dashboard
 ✅

---

### 3.6 References


Zylo Finance, 2025. How to assess loan grades in P2P platforms: a guide for Zylo P2P investment users. [online] Available at: https://zylofinance.in/how-to-assess-loan-grades-in-p2p-platforms-a-guide-for-zylo-p2p-investment-users/ [Accessed 16 November 2025].
Kwak, S.K. and Kim, J.H., 2017. Statistical data preparation: management of missing values and outliers in clinical data. Korean Journal of Anesthesiology. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC5548942/ [Accessed 16 November 2025].

Saunders, M., Lewis, P. and Thornhill, A. (2019) Research methods for business students. 8th edn. Harlow: Pearson Education.

Siddiqi, N., 2012. Credit risk scorecards: developing and implementing intelligent credit scoring. Hoboken, NJ: Wiley.


Brown, I. and Mues, C. (2012) 'An experimental comparison of classification algorithms for credit scoring', Decision Support Systems, 52(2), pp. 487-496.

Siddiqi, N. (2017) Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards. 2nd edn. Hoboken, NJ: John Wiley & Sons.

Scikit-learn (2023) Scikit-learn: Machine Learning in Python. Available at: https://scikit-learn.org (Accessed: 16 November 2025).
