#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>

#include "operation.h"

using namespace std;

class Operation;
class Variable;

// represents a computation graph
// this class abstracts a good interface for operations
// and runs the operations when given a output operation
// example:
/*
Graph g;
auto x = g.variable(2);
auto y = g.exp(x);
g.run(y);
cout << y->result << endl; // prints e^2
cout << x->grad << endl; // prints dy/dx
*/
class Graph {
private:
	// operations in the graph
	vector<Operation*> ops;

	// clear all the grads in operations
	// internal function dont use it outside
	void zero_grads();

public:
	// creates a operation of type Variable and returns a pointer to it
	Variable* variable(double value = 0);

	// all of the operation interface functions asks for Operation* parameter(s) 
	// then creates a new operation and returns its pointer
	Operation* add(Operation* op1, Operation* op2);
	Operation* sub(Operation* op1, Operation* op2);
	Operation* mul(Operation* op1, Operation* op2);
	Operation* div(Operation* op1, Operation* op2);
	Operation* log(Operation* op);
	Operation* exp(Operation* op);
	Operation* pow(Operation* op1, Operation* op2);
	Operation* sqrt(Operation* op);
	Operation* sin(Operation* op);
	Operation* cos(Operation* op);
	Operation* tan(Operation* op);
	Operation* asin(Operation* op);
	Operation* acos(Operation* op);
	Operation* atan(Operation* op);

	// run the graph given a output operation
	double run(Operation *output);
};

#endif