#include "doubleflow.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

/* 
trains a logistic regression model that learns the binary AND operation
*/
void logistic_regression() {
	mt19937 generator;
	std::normal_distribution<double> normal(0, 0.1);

	Graph g;

	const int m = 4, n = 2;

	Variable* x[m][n];
	Variable* r[m];

	// this is our training data
	x[0][0] = g.variable(0); x[0][1] = g.variable(0); r[0] = g.variable(0);
	x[1][0] = g.variable(0); x[1][1] = g.variable(1); r[1] = g.variable(0);
	x[2][0] = g.variable(1); x[2][1] = g.variable(0); r[2] = g.variable(0);
	x[3][0] = g.variable(1); x[3][1] = g.variable(1); r[3] = g.variable(1);


	// weights ans bias
	Variable* w[n];
	Variable* b = g.variable(0);

	// initialize weights normally distrubuted
	for (int i = 0; i < n; i++) {
		w[i] = g.variable(normal(generator));
	}
	
	// predictions
	Operation* y[m];

	for (int i = 0; i < m; i++) {
		y[i] = g.variable(0);

		// multiply the term with weights and add them
		for (int j = 0; j < n; j++) {
			y[i] = g.add(y[i], g.mul(w[j], x[i][j]));
		}

		// add the bias
		y[i] = g.add(b, y[i]);

		// sigmoid operation
		y[i] = g.div(g.variable(1), g.add(g.variable(1), g.exp(g.mul(g.variable(-1), y[i]))));
	}

	// calculate cross-entropy loss
	Operation* loss = g.variable(0);

	for (int i = 0; i < m; i++) {
		auto t1 = g.mul(g.mul(g.variable(-1), r[i]), g.log(y[i]));
		auto t2 = g.mul(g.sub(r[i], g.variable(1)), g.log(g.sub(g.variable(1), y[i])));

		loss = g.add(loss, g.add(t1, t2));
	}

	loss = g.div(loss, g.variable(m));

	// we defined our computation graph so far
	// now its time to train our model

	// we are using the gradient descent algorithm to train our parameters

	double lr = 10;

	cout << fixed << setprecision(4);
	for (int epoch = 1; epoch <= 1000; epoch++) {
		// calculate loss and gradients
		double cur_loss = g.run(loss);

		if (epoch % 10 == 0) {
			cout << "epoch=" << epoch << " loss=" << loss->result << endl;
		}

		// update parameters
		for (int i = 0; i < n; i++) {
			w[i]->set(w[i]->result - lr * w[i]->grad);
		}

		b->set(b->result - lr * b->grad);
	}

	cout << "w: " << w[0]->result << " " << w[1]->result << endl;
	cout << "b: " << b->result << endl;

	for (int i = 0; i < m; i++) {
		cout << i << "th prediction: " << y[i]->result << endl;
	}
}

int main() {
	logistic_regression();

	return 0;
}