# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- The model is a Random Forest classifier from the Sklearn library.
- The number of trees in the forest (n_estimators) is set to 100. 
- Rest of the hyperparamters are kept as default.

## Intended Use

Predict whether income exceeds $50K/yr based on census data.

## Training Data

- The [dataset](https://archive.ics.uci.edu/ml/datasets/census+income) contains information from the 1994 Census database.
- There is a total of 15 attributes including the target variable of interest: salary.

## Evaluation Data

- After cleaning the data, the data set has 30162 rows and 15 attributes. 
- A train-test-split of 80-20 was used. (80% train, 20% test) 

## Metrics

- Final Performance: Precision: 0.73, Recall: 0.64 and Fbeta: 0.68.

## Ethical Considerations

- Can take into considerations of sex and race disparities based on data or model performance.
- Whether there is bias or fairness in model.

## Caveats and Recommendations

- Further tune the hyperparameters such as using a grid search to improve performance.
- Try out a different model