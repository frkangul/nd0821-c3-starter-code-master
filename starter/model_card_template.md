# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
It is Adaboost classifier with default parameters in Scikit-learn library.
## Intended Use
It predicts the salary class of a customer with some financial and demographic features.
## Training Data
Data can be found on the below link.
https://archive.ics.uci.edu/ml/datasets/census+income
80% of the dataset is used for the training part.
## Evaluation Data
20% of the dataset is used for the evaluation part.
## Metrics
The model was evaluated using precision, recall, and fbeta scores. The values are around 0.76, 0.61, 0.68 respectively.

## Ethical Considerations
Dataset includes features like race, gender and origin country. This may drive our model to potentially discriminate people.
When you examine the data slice performance over gender, you will see that while recall for male gender is 0.85, recall for female is 0.65. Our model is discriminating females.
## Caveats and Recommendations
We can add more demographic features to know customers better. After carefull examination of data slice performance, we can add more data and work on some specific discriminated parts more.  