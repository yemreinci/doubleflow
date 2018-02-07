#include <math.h>
#include "operation.h"

Variable::Variable(double _value) : value(_value) {}

void Variable::forward() {
	this->result = this->value;
}

void Variable::backward() {}

void Variable::set(double value) {
	this->value = value;
}

void Add::forward() {
	this->result = (this->ins[0]->result) + (this->ins[1]->result);
}

void Add::backward() {
	this->ins[0]->grad += this->grad;
	this->ins[1]->grad += this->grad;
}

void Mul::forward() {
	this->result = (this->ins[0]->result) * (this->ins[1]->result);
}

void Mul::backward() {
	this->ins[0]->grad += (this->grad) * (this->ins[1]->result);
	this->ins[1]->grad += (this->grad) * (this->ins[0]->result);
}

void Div::forward() {
	this->result = (this->ins[0]->result) / (this->ins[1]->result);
}

void Div::backward() {
	double &x = this->ins[0]->result;
	double &y = this->ins[1]->result;
	this->ins[0]->grad += (this->grad) / y;
	this->ins[1]->grad += (this->grad) *  - x / (y * y);
}

void Log::forward() {
	this->result = log(this->ins[0]->result);
}

void Log::backward() {
	this->ins[0]->grad += (this->grad) / (this->ins[0]->result);
}

void Exp::forward() {
	this->result = exp(this->ins[0]->result);
}

void Exp::backward() {
	this->ins[0]->grad += (this->grad) * (this->result);
}

void Sin::forward() {
	this->result = sin(this->ins[0]->result);
}

void Sin::backward() {
	this->ins[0]->grad += (this->grad) * cos(this->ins[0]->result);
}

void Cos::forward() {
	this->result = cos(this->ins[0]->result);
}

void Cos::backward() {
	this->ins[0]->grad += (this->grad) * -sin(this->ins[0]->result);
}

void Asin::forward() {
	this->result = asin(this->ins[0]->result);
}

void Asin::backward() {
	double &x = this->ins[0]->result;
	this->ins[0]->grad += (this->grad) / sqrt(1 - x * x);
}

void Acos::forward() {
	this->result = acos(this->ins[0]->result);
}

void Acos::backward() {
	double &x = this->ins[0]->result;
	this->ins[0]->grad += (this->grad) / -sqrt(1 - x * x);
}

void Atan::forward() {
	this->result = atan(this->ins[0]->result);
}

void Atan::backward() {
	double &x = this->ins[0]->result;
	this->ins[0]->grad += (this->grad) / (1 + x * x);
}