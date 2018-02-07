#ifndef _OPERATION_H_
#define _OPERATION_H_

#include <vector>
#include "graph.h"

using namespace std;

// represents a single operation on the computation graph
// multiplication, cosinus... etc.
class Operation {
friend class Graph;

protected:
	// inputs of the operation
	vector<Operation*> ins;
	// forward calculation of the operator
	virtual void forward() = 0;
	// derivative calculation of the operator
	virtual void backward() = 0;

public:
	double result;
	double grad;
};

class Variable: public Operation {
private:
	double value;

	void forward();
	void backward();

public:
	Variable(double _value);
	void set(double value);
};

class Add: public Operation {
private:
	void forward();
	void backward();
};

class Mul: public Operation {
private:
	void forward();
	void backward();
};

class Div: public Operation {
private:
	void forward();
	void backward();
};

class Log: public Operation {
private:
	void forward();
	void backward();
};

class Exp: public Operation {
private:
	void forward();
	void backward();
};

class Sin: public Operation {
private:
	void forward();
	void backward();
};

class Cos: public Operation {
private:
	void forward();
	void backward();
};

class Asin: public Operation {
private:
	void forward();
	void backward();
};

class Acos: public Operation {
private:
	void forward();
	void backward();
};

class Atan: public Operation {
private:
	void forward();
	void backward();
};

#endif