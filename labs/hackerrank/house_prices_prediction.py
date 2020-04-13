"""
Link to question: https://www.hackerrank.com/challenges/predicting-house-prices/problem

Task
----------------
Charlie wants to buy a house. He does a detailed survey of the area where he wants to live, in which he quantifies, normalizes, and maps the desirable features of houses to values on a scale of 0 to 1 so the data can be assembled into a table. If Charlie noted F features, each row contains F space-separated values followed by the house price in dollars per square foot (making for a total of  columns). If Charlie makes observations about H houses, his observation table has H rows. This means that the table has a total of (F+1) x H entries.

Unfortunately, he was only able to get the price per square foot for certain houses and thus needs your help estimating the prices of the rest! Given the feature and pricing data for a set of houses, help Charlie estimate the price per square foot of the houses for which he has compiled feature data but no pricing.

Important Observation: The prices per square foot form an approximately linear function for the features quantified in Charlie's table. For the purposes of prediction, you need to figure out this linear function.

Recommended Technique: Use a regression-based technique. At this point, you are not expected to account for bias and variance trade-offs.

Input Format
----------------
The first line contains 2 space-separated integers, F(the number of observed features) and N(the number of rows/houses for which Charlie has noted both the features and price per square foot).
The N subsequent lines each contain F+1 space-separated floating-point numbers describing a row in the table; the first F elements are the noted features for a house, and the very last element is its price per square foot.

The next line (following the table) contains a single integer, T, denoting the number of houses for for which Charlie noted features but does not know the price per square foot.
The T subsequent lines each contain F space-separated floating-point numbers describing the features of a house for which pricing is not known.

Constraints
----------------
-> 1 <= F <= 10
-> 5 <= N <= 100
-> 1 <= T <= 100
-> 0 <= Price Per Square Foot <= 10^6
-> 0 <= Factor Values <= 1

Scoring
----------------
For each test case, we will compute the following:
-> d = Normalized Distance from Expected answer = abs(Computed-Expected)/Expected

There are multiple ways to approach this problem that account for bias, variance, various subjective factors, and "noise". We take a realistic approach to scoring and permit up to a  swing of our expected answer.

-> d_adjusted = max(d - 0.1, 0)
-> Score for each test case === max(1 - d_adjusted,0)
-> Score for the test case === (Average score for all the tests if contains) x M, where M is the maximum possible score for the test case.

Consider a test case in which we only need to find the pricing for  house. Suppose our expected answer is, and your answer is:
d = (10 - 9.5)/10 = 0.05
d_adjusted = max(0.05 - 0.1, 0) = 0

The score for a test case with 10 points = max(1,0) x 10 = 10

Output Format
----------------
Print  lines, where each line  contains the predicted price for the  house (from the second table of houses with unknown prices per square foot).

Sample Input
----------------
2 7
0.18 0.89 109.85
1.0 0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77
4
0.49 0.18
0.57 0.83
0.56 0.64
0.76 0.18

Sample Output
----------------
105.22
142.68
132.94
129.71
"""

# Enter your code here. Read input from STDIN. Print output to STDOUT
import pandas as pd
from sklearn import linear_model

def main():
    train_info = input()
    num_features = int(train_info.split(' ')[0])
    num_train = int(train_info.split(' ')[1])

    X_column_names = ['f'+str(i) for i in range(1, num_features+1)]
    y_column_names = ['target']
    X_train_arr = []
    y_train_arr = []
    for idx in range(num_train):
        new_data_str = input()
        new_data_arr = new_data_str.split(' ')
        X_train_arr.append(new_data_arr[:-1])
        y_train_arr.append(new_data_arr[-1])

    X_train = pd.DataFrame(X_train_arr, columns=X_column_names)
    y_train = pd.DataFrame(y_train_arr, columns=y_column_names)

    test_info = input()
    num_test = int(test_info)

    X_test_arr = []
    for idx in range(num_test):
        new_data_str = input()
        new_data_arr = new_data_str.split(' ')
        X_test_arr.append(new_data_arr)

    X_test = pd.DataFrame(X_test_arr, columns=X_column_names)

    # Modeling
    lm = linear_model.LinearRegression(
        fit_intercept=True,
        normalize=True
    )
    lr_model = lm.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_pred_arr = [pred[0] for pred in y_pred]

    # Print predictions
    for idx in range(len(y_pred_arr)):
        print(y_pred_arr[idx])


main()