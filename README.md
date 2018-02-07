# DoubleFlow

**DoubleFlow** is an automatic differentiation library. It is highly inspired by [TensorFlow](https://github.com/tensorflow/tensorflow), but it only works with usual double variables, not tensors :(  

It implements 14 mathematical operations:
  * add
  * sub
  * mul
  * div
  * log
  * exp
  * pow
  * sqrt
  * sin
  * cos
  * tan
  * asin
  * acos
  * atan

## Example

``` c++
// you first need to declare a Graph variable
Graph g;

auto x1 = g.variable(3);  // creates a variable with initial value 3
auto x2 = g.variable(2);

auto y = g.mul(x1, x2); // creates a variable such that y = x1 * x2

g.run(y); // runs the graph with y as the output

cout << y->result << endl; // prints 6
cout << x1->grad << endl; // prints dy/dx1, which is equal to x2

// you can update the variables and run the graph again

x1->set(4);
x2->set(0.5);

g.run(y);

cout << y->result << endl;
cout << x1->grad << endl; 
```

There is also a [demo](demo.cpp) which trains a basic logistic regression model.