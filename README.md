# Mooc-dropoff-prediction

I used 4 kaggle datasets with MOOC course drop off information to predict MOOC dropoff probabilities.

4 input dataset:

The first one is “enrollment list”, the columns of which are “enrollment_id”, “course_id” and “user_id”, which show the information that a user enrolled in a course with a specific enrollment id.
The second file is “activity log”, which records each enrollment’s study event and happen time.
The third table “train_label” has its first column enrollment id, and the other column is “dropout_prob”, in which the value of “1” means the user dropped this course, while “0” means the user did not.
The last one is “sample_submission” which gives a sample of what my result will be look like. The second column of “dropout_prob” is some float number rather than a series of “0” s or “1” s, which means I will give some probabilities of “dropout_prob” equals to “1”, that is, the probability of a person dropped a specific course.


Model used:

Logistic regression, KNN, Random forest classifier, Gradient boosting, voting function etc.
