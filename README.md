# Scope

The project scope is to implement selected data processing steps in the
context of a full ML pipeline, and demonstrate the impact on a
pre-defined metric using two prediction tasks. The implementation focus
is on the data preparation prior to training - outlier removal and class
balancing, and also on a model monitoring step - drift detection.

# Prediction tasks

The two binary classification tasks:

1.  Bank Marketing - classifying whether a bank customer is likely to
    subscribe to a long term deposit following a direct marketing
    approach. The data used is the [Bank marketing
    dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
    The data contains 4119 examples, 17 features.

2.  Credit Risk - classifying bank customers as low or high credit risk.
    The data used is the [Credit risk
    dataset](https://archive.ics.uci.edu/ml/datasets/statlog+\(german+credit+data\)).
    The data contains 1000 examples, 20 features.

# Metrics

  - The qualitative objectives of the project are improving the per
    class classification accuracy in both prediction tasks.

  - The quantifiable metric is the per-class F1, Precision and Recall.

  - the bank marketing task suffers from high rate of false negatives
    (baseline value is 63%). Reducing the FNR promotes addressing more
    clients which are likely to purchase the product and thus may
    increase the client’s revenue.

  - The credit predicting task suffers from high FPR (baseline value is
    61%). Reducing FPR reduces probability of giving high risk credit
    and thus reduce the client’s risk.

  - The metrics will be evaluated by holding out test sets from both
    datasets. Metrics will be evaluated on these sets.

# Pipeline Architecture

The data pipeline is shown in figure [1](#fig:fig1) and includes the
following steps:

![ML Pipeline](Data%20Processing%20Pipeline.png)

1.  Data pre-process:
    
      - Splitting to train & test sets.
    
      - Feature transformation (encoding categorical feature as one-hot
        etc.)
    
      - Scaling: Fitting a scaler to the train set (the test set will be
        transformed with the parameters learned from the train set).

2.  Anomaly detection - This step will remove outliers from the training
    set prior to the next step which is balancing. Since balancing the
    data involves re-sampling/synthesizing new examples from existing
    ones, it might amplify any noise which exists in the data. Hence,
    the goal here is to clean the data from this noise (as an added
    value, removing outliers often helps the model fitting itself). The
    pipeline will support two anomaly detection algorithms:
    
    1.  Isolation forest ()
    
    2.  One-class SVM ()
    
    Since for highly imbalanced data sets, all the minority class
    samples might be classified as anomalies, we will perform the
    anomaly detection per class. We will also place a threshold (which
    is a pipeline hyper-parameter) on the acceptable data loss (maximal
    percentage of samples to discard) from each class. A grid search is
    performed to select the anomaly detection algorithm and tune its
    parameters. We will examine two strategies for auto-tuning:
    
    1.  Heuristic based on fairness (discarding the same percentage of
        examples from each class) and aggressiveness (discarding a
        number of examples close to the allowed threshold).
    
    2.  Performance based - maximizing the per-class F1 score using
        cross-validation on the model architecture which is fed to the
        pipeline.

3.  Train set resampling - This step’s goal is to balance the training
    set. The resampler will utilize the SMOTE () algorithm to generate
    new synthetic samples from the minority class. Since over-sampling a
    highly imbalanced set might defeat the purpose, a tuning is needed
    to decide how much to over-sample and if down-sampling of the
    majority class is also needed. We will examine two strategies for
    auto-tuning:
    
    1.  Heuristic based on class size ratio.
    
    2.  Performance based - maximizing the per-class F1 score using
        cross-validation on the model architecture which is fed to the
        pipeline.

4.  Training & Evaluation - in these steps the model architecture (which
    is an input to the pipeline) is fitted to the training set and then
    evaluated on the test set, producing the objective metric defined in
    section 3.

5.  CDD - Concept drift detector. This step will use the test data as a
    reference, and will detect a concept drift between this data and a
    new batch of classification data. The detection is done by applying
    a statistical test (Kolmogorov-Smirnov) to see if the response (i.e.
    the soft prediction of the model) distributions between reference
    and new data differ significantly.
