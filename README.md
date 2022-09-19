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

   
   
   
