# ft_linear_regression
This project is from 42 school.  
This is the introduction to machine learning with the concept of [linear regression](https://en.wikipedia.org/wiki/Linear_regression) using the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm.  

The objective of this is to create a program that will be able to predict the price of a car based on its age (in km).  
The program will be trained using a provided dataset.  

> [!INFO]
> Every mathematical notion is written using [LaTeX](https://fr.wikipedia.org/wiki/LaTeX)

## Theorical
To create this program we need to create a fonction that based on the dataset will calculate the best average price based on the car age.  

<table>
    <tr>
        <td>
            <p>If we represent the dataset it's something like this :</p>
            <img src="doc/dataset.png" width=500>
        </td>
        <td>
            <p>And we want to obtain something like this :</p>
            <img src="doc/dataset_draw.png" width=500>
        </td>
    </tr>
</table>

As you see this representation of our "solution" is a straight line, which mean that it is defined by a linear function.  
In other words a linear function equals $a*x+b$.  

Where **a is the slope of the function** and **b is the intercept**.  

### Cost function
To find the best values of our linear fonction, we are going to make predictions and compare it to the actual value (dataset).  
We will calculate the distance between the two points:  
This is the euclidian distance formula $\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$, since we will only use the Y axis we do not care about X, it can be simplified to $\sqrt{(y_2-y_1)^2}$.  

We also do not need to keep the **sqrt** to reduce cost of calculation.
It will be the square error defined by $(y_2-y_1)^2$

Now we have the row formula but we might replace those values by the real ones:  $(f(x_i) - y_i)^2$  
Where $f(x_i)$ represent the "predicted value" and $y_i$ the real one.  

This is the error for $i$, but now we need to to calculate the sum of all the erros made by the program for all the dataset (for each points).  
In maths, it is represented by : $\frac{1}{2m}\sum_{i=0}^m (f(x_i) - y_i)^2$  

Let me explain, the sigma symbol says that it will iterate over each $i$, which start to 0 until it reach $m$ that represent the lenght of our dataset.  
The $\frac{1}{2m}$ is used to have average of the errors, the 2 is here to help us later (for the derivative calculation).  
Congratulation you have you have discovered the mean square error formula.  

### Gradient descent
If we replace our mean square error formula using our linear fonction, it will look like this: $J(a,b)=\frac{1}{2m}\sum_{i=0}^m (a*x_i+b - y_i)^2$

This is a convex function. 
<img src="doc/convexe.png" width=500>

We want to minimize our cost to have the best result for our prediction program, so we need to find a minimum. There are multiples ways to minimize a cost function  
In this project it is specified to use the **gradient descent algorithm**.

#### Derivative function
Just a little remember about what is a it.  
A **derivative function** is a tool that we can use in math to mesure how fast a function is changing at a given point. If the derivative is positive it mean that the fonction is going up, else it mean that is going down. This calculate the **slope of a fonction** at a precise point.  
A derivate function is written with $f'(x)$ or $\frac{df}{dx}$.  

#### Iterative process
$a_{i+1}=a_i-\alpha*\frac{1}{m}\sum_{i=0}^m x*(a*x_i+b - y_i)$

$b_{i+1}=b_i-\alpha*\frac{1}{m}\sum_{i=0}^m (a*x_i+b - y_i)$

### Normalization


## Pratical

### How to use it

### Matplotlib

## Resources

- [Machine Learnia](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)
- [Derivative Function - FR](https://www.youtube.com/watch?v=9Mann4wOGJA&list=PLVUDmbpupCaoY7qihLa2dHc9-rBgVrgWJ)
- [Derivative Composated Functions - FR](https://www.youtube.com/watch?v=lwcFgnbs0Ew)
- [Suites - FR](https://www.youtube.com/watch?v=8I6dotcdW3I)
- [Derivative Function Easy - FR](https://www.youtube.com/watch?v=RLEE-iSBimc)