"""
Link to question: https://www.hackerrank.com/challenges/predicting-office-space-price/problem

The Problem
----------------
Charlie wants to purchase office-space. He does a detailed survey of the offices and corporate complexes in the area, and tries 
to quantify a lot of factors, such as the distance of the offices from residential and other commercial areas, schools and 
workplaces; the reputation of the construction companies and builders involved in constructing the apartments; the distance of 
the offices from highways, freeways and important roads; the facilities around the office space and so on.

Each of these factors are quantified, normalized and mapped to values on a scale of 0 to 1. Charlie then makes a table. Each 
row in the table corresponds to Charlie's observations for a particular house. If Charlie has observed and noted F features, 
the row contains F values separated by a single space, followed by the office-space price in dollars/square-foot. If Charlie 
makes observations for H houses, his observation table has (F+1) columns and H rows, and a total of (F+1) * H entries.

Charlie does several such surveys and provides you with the tabulated data. At the end of these tables are some rows which have 
just F columns (the price per square foot is missing). Your task is to predict these prices. F can be any integer number between 
1 and 5, both inclusive.

There is one important observation which Charlie has made.

The prices per square foot, are (approximately) a polynomial function of the features in the observation table. 
This polynomial always has an order less than 4

Input Format
----------------
The first line contains two space separated integers, F and N. Over here, F is the number of observed features. 
N is the number of rows for which features as well as price per square-foot have been noted.
This is followed by a table having F+1 columns and N rows with each row in a new line and each column separated by a single space.
The last column is the price per square foot.

The table is immediately followed by integer T followed by T rows containing F columns.

Constraints
----------------
1 <= F <= 5
5 <= N <= 100
1 <= T <= 100
0 <= Price Per Square Foot <= 10^6 0 <= Factor Values <= 1

Recommended Technique
----------------
Use a regression based technique. At this point, you are not expected to account for bias and variance trade-offs.

Scoring
----------------
For each test in a test case file we will compute the following:

d = Normalized Distance from expected answer  
  = abs(Computed-Expected)/Expected  
Since there can be multiple ways to approach this problem, which account for bias, variance, various subjective factors and "noise",
we will take a realistic approach to scoring, and permit upto +/- 10% swing of our expected answer.

d_adjusted = max(d - 0.1, 0)  
Score for each test = max(1-d_adjusted,0)
Score for the test case file == (Average of the scores for the tests it contains) * M
Where M is the Max possible score for the test case.

Suppose we have a test case file with just one test. Suppose our expected answer is 10
And your answer is: 9.5
d = (10 - 9.5) / 10 = 0.05
d_adjusted = max(0.05 - 0.1,0) = 0 Score = max(1-d_adjusted, 0) = max(1,0) = 1

Output Format
----------------
T lines. Each line 'i' contains the predicted price for the 'i'th test case.

Sample Input
----------------
2 4
0.44 0.68 511.14
0.99 0.23 717.1
0.84 0.29 607.91
0.28 0.45 270.4
4
0.05 0.54
0.91 0.91
0.31 0.76
0.51 0.31

Sample Output
----------------
180.38
1312.07
440.13
343.72
"""


# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def main():
    num_features, num_train = [int(i) for i in input().split()]
    Xy_train = np.array([input().split(' ') for _ in range(num_train)], dtype=float)
    
    num_test = int(input())
    X_test = np.array([input().split(' ') for _ in range(num_test)], dtype=float)

    # Modeling
    lm = linear_model.LinearRegression()
    poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
    lm.fit(
        poly.fit_transform(Xy_train[:, :-1]), Xy_train[:, -1]
    )

    y_pred = lm.predict(
        poly.fit_transform(X_test)
    )
    print(*y_pred, sep="\n")

main()