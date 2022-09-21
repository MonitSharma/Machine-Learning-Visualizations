# Visualizing and Implementing various Machine Learning Algorithms

With the use of machine learning (ML), which is a form of artificial intelligence (AI), software programmes can predict outcomes more accurately without having to be explicitly instructed to do so. In order to forecast new output values, machine learning algorithms use historical data as input.

Machine learning is frequently used in recommendation engines. Business process automation (BPA), predictive maintenance, spam filtering, malware threat detection, and fraud detection are a few additional common uses.


----

## Why is Machine Learning Important?

Machine learning is significant because it aids in the development of new goods and provides businesses with a picture of trends in consumer behaviour and operational business patterns. A significant portion of the operations of many of today's top businesses, like Facebook, Google, and Uber, revolve around machine learning. For many businesses, machine learning has emerged as a key competitive differentiation.

----

# Contents

## Gradient Descent
---
An optimization approach called gradient descent is frequently used to train neural networks and machine learning models.
These models gain knowledge over time by using training data, and the cost function in gradient descent especially serves as a barometer by
assessing the accuracy of each iteration of parameter changes. The model will keep changing its parameters to provide the minimal error until
the function is close to or equal to zero. Machine learning models can be effective tools for applications in artificial intelligence (AI) and
computer science after their accuracy has been adjusted.
---

## How does gradient descent work?
---
Before we dive into gradient descent, it may help to review some concepts from linear regression. You may recall the following formula for the slope of a line, which is $y = mx + b$, where $m$ represents the slope and $b$ is the intercept on the y-axis.

You may also recall plotting a scatterplot in statistics and finding the line of best fit, which required calculating the error between the actual output and the predicted output (y-hat) using the mean squared error formula. The gradient descent algorithm behaves similarly, but it is based on a convex function, such as the one below:

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/images/ICLH_Diagram_Batch_01_04-GradientDescent-WHITEBG_0.png)

---

The starting point is just an arbitrary point for us to evaluate the performance. From that starting point, we will find the derivative (or slope), and from there, we can use a tangent line to observe the steepness of the slope. The slope will inform the updates to the parameters—i.e. the weights and bias. The slope at the starting point will be steeper, but as new parameters are generated, the steepness should gradually reduce until it reaches the lowest point on the curve, known as the point of convergence.   

Similar to finding the line of best fit in linear regression, the goal of gradient descent is to minimize the cost function, or the error between predicted and actual y. In order to do this, it requires two data points—a direction and a learning rate. These factors determine the partial derivative calculations of future iterations, allowing it to gradually arrive at the local or global minimum (i.e. point of convergence). More detail on these components can be found below:

+ Learning Rate : (also referred to as step size or the alpha) is the size of the steps that are taken to reach the minimum. This is typically a small value, and it is evaluated and updated based on the behavior of the cost function. High learning rates result in larger steps but risks overshooting the minimum. Conversely, a low learning rate has small step sizes. While it has the advantage of more precision, the number of iterations compromises overall efficiency as this takes more time and computations to reach the minimum.

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/images/Learning%20Rate.jpg)

---

## Challenges with gradient descent

While gradient descent is the most common approach for optimization problems, it does come with its own set of challenges. Some of them include:
---

### Local minima and saddle points
For convex problems, gradient descent can find the global minimum with ease, but as nonconvex problems emerge, gradient descent can struggle to find the global minimum, where the model achieves the best results.

Recall that when the slope of the cost function is at or close to zero, the model stops learning. A few scenarios beyond the global minimum can also yield this slope, which are local minima and saddle points. Local minima mimic the shape of a global minimum, where the slope of the cost function increases on either side of the current point. However, with saddle points, the negative gradient only exists on one side of the point, reaching a local maximum on one side and a local minimum on the other. Its name inspired by that of a horse’s saddle.

Noisy gradients can help the gradient escape local minimums and saddle points

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/images/Local%20Minimum_SaddlePoints.jpg)


### Vanishing and Exploding Gradients
In deeper neural networks, particular recurrent neural networks, we can also encounter two other problems when the model is trained with gradient descent and backpropagation.

+ Vanishing gradients: This occurs when the gradient is too small. As we move backwards during backpropagation, the gradient continues to become smaller, causing the earlier layers in the network to learn more slowly than later layers. When this happens, the weight parameters update until they become insignificant—i.e. 0—resulting in an algorithm that is no longer learning.
+ Exploding gradients: This happens when the gradient is too large, creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN. One solution to this issue is to leverage a dimensionality reduction technique, which can help to minimize complexity within the model.

---

## Implementation and Visualization in Python

The code implementation can be found here:

+ [Gradient Descent in 2-d](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/algorithms/gradient_descent_2d.py)

### Visualization 

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/GradientDescent2D.gif)
 ---  
+ [Gradient Descent in 2-d with Momentum](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/algorithms/momentum_2d.py)

### Visualization

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/GradientDescentWithMomentum2D.gif)
   ----
+ [Gradient Descent in 3-d](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/algorithms/gradient_descent_3d.py)


### Visualization

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/GradientDescent3D.gif)
----
+ [Gradient Descent in 3-d with Momentum](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/algorithms/momentum_3d.py)

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/GradientDescentWithMomentum3D.gif)

   
   -----
   
   
 ## Linear Regression
 
 
 Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.

This form of analysis estimates the coefficients of the linear equation, involving one or more independent variables that best predict the value of the dependent variable. Linear regression fits a straight line or surface that minimizes the discrepancies between predicted and actual output values. There are simple linear regression calculators that use a “least squares” method to discover the best-fit line for a set of paired data. You then estimate the value of X (dependent variable) from Y (independent variable).

----

### Why linear regression is important?

Linear-regression models are relatively simple and provide an easy-to-interpret mathematical formula that can generate predictions. Linear regression can be applied to various areas in business and academic study.

You’ll find that linear regression is used in everything from biological, behavioral, environmental and social sciences to business. Linear-regression models have become a proven way to scientifically and reliably predict the future. Because linear regression is a long-established statistical procedure, the properties of linear-regression models are well understood and can be trained very quickly.



-----

### Key assumptions of effective linear regression

Assumptions to be considered for success with linear-regression analysis:

+ For each variable: Consider the number of valid cases, mean and standard deviation. 
+ For each model: Consider regression coefficients, correlation matrix, part and partial correlations, multiple R, R2, adjusted R2, change in R2, standard error of the estimate, analysis-of-variance table, predicted values and residuals. Also, consider 95-percent-confidence intervals for each regression coefficient, variance-covariance matrix, variance inflation factor, tolerance, Durbin-Watson test, distance measures (Mahalanobis, Cook and leverage values), DfBeta, DfFit, prediction intervals and case-wise diagnostic information. 
+ Plots: Consider scatterplots, partial plots, histograms and normal probability plots.
+ Data: Dependent and independent variables should be quantitative. Categorical variables, such as religion, major field of study or region of residence, need to be recoded to binary (dummy) variables or other types of contrast variables.  
+ Other assumptions: For each value of the independent variable, the distribution of the dependent variable must be normal. The variance of the distribution of the dependent variable should be constant for all values of the independent variable. The relationship between the dependent variable and each independent variable should be linear and all observations should be independent.


-----

### Examples of linear-regression success

#### Evaluating trends and sales estimates
You can also use linear-regression analysis to try to predict a salesperson’s total yearly sales (the dependent variable) from independent variables such as age, education and years of experience.

#### Analyze pricing elasticity
Changes in pricing often impact consumer behavior — and linear regression can help you analyze how. For instance, if the price of a particular product keeps changing, you can use regression analysis to see whether consumption drops as the price increases. What if consumption does not drop significantly as the price increases? At what price point do buyers stop purchasing the product? This information would be very helpful for leaders in a retail business.

#### Assess risk in an insurance company
Linear regression techniques can be used to analyze risk. For example, an insurance company might have limited resources with which to investigate homeowners’ insurance claims; with linear regression, the company’s team can build a model for estimating claims costs. The analysis could help company leaders make important business decisions about what risks to take.

#### Sports analysis
Linear regression isn’t always about business. It’s also important in sports. For instance, you might wonder if the number of games won by a basketball team in a season is related to the average number of points the team scores per game. A scatterplot indicates that these variables are linearly related. The number of games won and the average number of points scored by the opponent are also linearly related. These variables have a negative relationship. As the number of games won increases, the average number of points scored by the opponent decreases. With linear regression, you can model the relationship of these variables. A good model can be used to predict how many games teams will win.



----

## Visualization
### Linear Regression
![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/LinearRegression.gif)

---
### Linear Regression with Non-Linear Dataset

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/NonLinearLinearRegression.gif)


---

## Logistic Regression 

his type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. In logistic regression, a logit transformation is applied on the odds—that is, the probability of success divided by the probability of failure. This is also commonly known as the log odds, or the natural logarithm of odds, and this logistic function is represented by the following formulas:

$$ \log (\pi) = 1/(1+ \exp(-\pi)) $$

$$ \log (\pi/(1-\pi)) = \beta_0 + \beta_1*X_1 + … + B_k*K_k $$

In this logistic regression equation, logit(pi) is the dependent or response variable and x is the independent variable. The beta parameter, or coefficient, in this model is commonly estimated via maximum likelihood estimation (MLE). This method tests different values of beta through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimate. Once the optimal coefficient (or coefficients if there is more than one independent variable) is found, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability. For binary classification, a probability less than .5 will predict 0 while a probability greater than 0 will predict 1.  After the model has been computed, it’s best practice to evaluate the how well the model predicts the dependent variable, which is called goodness of fit. The Hosmer–Lemeshow test is a popular method to assess model fit.

---

## Types of logistic regression
There are three types of logistic regression models, which are defined based on categorical response.

+ Binary logistic regression: In this approach, the response or dependent variable is dichotomous in nature—i.e. it has only two possible outcomes (e.g. 0 or 1). Some popular examples of its use include predicting if an e-mail is spam or not spam or if a tumor is malignant or not malignant. Within logistic regression, this is the most commonly used approach, and more generally, it is one of the most common classifiers for binary classification.
+ Multinomial logistic regression: In this type of logistic regression model, the dependent variable has three or more possible outcomes; however, these values have no specified order.  For example, movie studios want to predict what genre of film a moviegoer is likely to see to market films more effectively. A multinomial logistic regression model can help the studio to determine the strength of influence a person's age, gender, and dating status may have on the type of film that they prefer. The studio can then orient an advertising campaign of a specific movie toward a group of people likely to go see it.
+ Ordinal logistic regression: This type of logistic regression model is leveraged when the response variable has three or more possible outcome, but in this case, these values do have a defined order. Examples of ordinal responses include grading scales from A to F or rating scales from 1 to 5. 


----

## Logistic regression and machine learning
Within machine learning, logistic regression belongs to the family of supervised machine learning models. It is also considered a discriminative model, which means that it attempts to distinguish between classes (or categories). Unlike a generative algorithm, such as naïve bayes, it cannot, as the name implies, generate information, such as an image, of the class that it is trying to predict (e.g. a picture of a cat).

Previously, we mentioned how logistic regression maximizes the log likelihood function to determine the beta coefficients of the model. This changes slightly under the context of machine learning. Within machine learning, the negative log likelihood used as the loss function, using the process of gradient descent to find the global maximum. This is just another way to arrive at the same estimations discussed above.

Logistic regression can also be prone to overfitting, particularly when there is a high number of predictor variables within the model. Regularization is typically used to penalize parameters large coefficients when the model suffers from high dimensionality.

Scikit-learn provides valuable documentation to learn more about the logistic regression machine learning model.

---

## Use cases of logistic regression
Logistic regression is commonly used for prediction and classification problems. Some of these use cases include:

+ Fraud detection: Logistic regression models can help teams identify data anomalies, which are predictive of fraud. Certain behaviors or characteristics may have a higher association with fraudulent activities, which is particularly helpful to banking and other financial institutions in protecting their clients. SaaS-based companies have also started to adopt these practices to eliminate fake user accounts from their datasets when conducting data analysis around business performance.
+ Disease prediction: In medicine, this analytics approach can be used to predict the likelihood of disease or illness for a given population. Healthcare organizations can set up preventative care for individuals that show higher propensity for specific illnesses.
+ Churn prediction: Specific behaviors may be indicative of churn in different functions of an organization. For example, human resources and management teams may want to know if there are high performers within the company who are at risk of leaving the organization; this type of insight can prompt conversations to understand problem areas within the company, such as culture or compensation. Alternatively, the sales organization may want to learn which of their clients are at risk of taking their business elsewhere. This can prompt teams to set up a retention strategy to avoid lost revenue.

---

## Visualization
### Logistic Regression
![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/DecisionBoundary.gif)

---

### Non Linear Training Set

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/NonLinearTrainingData.png)

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/NonLinearDecisionBoundary.gif)

---

## K-Nearest Neigbours

The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.


For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. While this is technically considered “plurality voting”, the term, “majority vote” is more commonly used in literature. The distinction between these terminologies is that “majority voting” technically requires a majority of greater than 50%, which primarily works when there are only two categories. When you have multiple classes—e.g. four categories, you don’t necessarily need 50% of the vote to make a conclusion about a class; you could assign a class label with a vote of greater than 25%.


## Compute KNN: distance metrics
To recap, the goal of the k-nearest neighbor algorithm is to identify the nearest neighbors of a given query point, so that we can assign a class label to that point. In order to do this, KNN has a few requirements:

### Determine your distance metrics

In order to determine which data points are closest to a given query point, the distance between the query point and the other data points will need to be calculated. These distance metrics help to form decision boundaries, which partitions query points into different regions. You commonly will see decision boundaries visualized with Voronoi diagrams.

While there are several distance measures that you can choose from, this article will only cover the following:

+ Euclidean distance (p=2): This is the most commonly used distance measure, and it is limited to real-valued vectors. Using the below formula, it measures a straight line between the query point and the other point being measured.

$$ d(x,y) = \sqrt{\sum_{i=1}^n  (y_i - x_i)^2} $$

+ Manhattan distance (p=1): This is also another popular distance metric, which measures the absolute value between two points. It is also referred to as taxicab distance or city block distance as it is commonly visualized with a grid, illustrating how one might navigate from one address to another via city streets

$$ d(x,y) = \bigg( \sum_{i=1}^m |x_i - y_i| \bigg) $$

+ Minkowski distance: This distance measure is the generalized form of Euclidean and Manhattan distance metrics. The parameter, p, in the formula below, allows for the creation of other distance metrics. Euclidean distance is represented by this formula when p is equal to two, and Manhattan distance is denoted with p equal to one

$$ d(x,y) = \bigg( \sum_{i=1}^n |x_i - y_i|) \bigg)^{1/p}$$

+ Hamming distance: This technique is used typically used with Boolean or string vectors, identifying the points where the vectors do not match. As a result, it has also been referred to as the overlap metric. This can be represented with the following formula:

$$ D_H = \bigg(\sum_{i=1}^k |x_i - y_i|) \bigg)$$

---

## Applications of k-NN in machine learning
The k-NN algorithm has been utilized within a variety of applications, largely within classification. Some of these use cases include:

- Data preprocessing: Datasets frequently have missing values, but the KNN algorithm can estimate for those values in a process known as missing data imputation.

- Recommendation Engines: Using clickstream data from websites, the KNN algorithm has been used to provide automatic recommendations to users on additional content. This research (link resides outside of ibm.com) shows that the a user is assigned to a particular group, and based on that group’s user behavior, they are given a recommendation. However, given the scaling issues with KNN, this approach may not be optimal for larger datasets.

- Finance: It has also been used in a variety of finance and economic use cases. For example, one paper (PDF, 391 KB)  (link resides outside of ibm.com) shows how using KNN on credit data can help banks assess risk of a loan to an organization or individual. It is used to determine the credit-worthiness of a loan applicant. Another journal (PDF, 447 KB)(link resides outside of ibm.com) highlights its use in stock market forecasting, currency exchange rates, trading futures, and money laundering analyses.

- Healthcare: KNN has also had application within the healthcare industry, making predictions on the risk of heart attacks and prostate cancer. The algorithm works by calculating the most likely gene expressions.

- Pattern Recognition: KNN has also assisted in identifying patterns, such as in text and digit classification (link resides outside of ibm.com). This has been particularly helpful in identifying handwritten numbers that you might find on forms or mailing envelopes.
   
   
   ---
   
   
## Advantages and disadvantages of the KNN algorithm
Just like any machine learning algorithm, k-NN has its strengths and weaknesses. Depending on the project and application, it may or may not be the right choice.

### Advantages
- Easy to implement: Given the algorithm’s simplicity and accuracy, it is one of the first classifiers that a new data scientist will learn.
- Adapts easily: As new training samples are added, the algorithm adjusts to account for any new data since all training data is stored into memory.

- Few hyperparameters: KNN only requires a k value and a distance metric, which is low when compared to other machine learning algorithms.

### Disadvantages
- Does not scale well: Since KNN is a lazy algorithm, it takes up more memory and data storage compared to other classifiers. This can be costly from both a time and money perspective. More memory and storage will drive up business expenses and more data can take longer to compute. While different data structures, such as Ball-Tree, have been created to address the computational inefficiencies, a different classifier may be ideal depending on the business problem.

- Curse of dimensionality: The KNN algorithm tends to fall victim to the curse of dimensionality, which means that it doesn’t perform well with high-dimensional data inputs. This is sometimes also referred to as the peaking phenomenon (PDF, 340 MB) (link resides outside of ibm.com), where after the algorithm attains the optimal number of features, additional features increases the amount of classification errors, especially when the sample size is smaller.

- Prone to overfitting: Due to the “curse of dimensionality”, KNN is also more prone to overfitting. While feature selection and dimensionality reduction techniques are leveraged to prevent this from occurring, the value of k can also impact the model’s behavior. Lower values of k can overfit the data, whereas higher values of k tend to “smooth out” the prediction values since it is averaging the values over a greater area, or neighborhood. However, if the value of k is too high, then it can underfit the data. 


---
## Visualizations

### K Nearest Neighbours 2-D

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/TopFeatures2D.png)


---

### K Nearest Neighborrs 3-D

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/TopFeatures3D.gif)


---

### K Means 2-D

![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/Clusters2D.png)

---

### K Means 3-D
![alt text](https://github.com/MonitSharma/Machine-Learning-Visualizations/blob/main/gifs/Clusters3D.gif)
