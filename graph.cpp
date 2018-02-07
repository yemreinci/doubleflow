#include "graph.h"

#include <map>
#include <vector>

Variable* Graph::variable(double value) {
	auto new_op = new Variable(value);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::add(Operation* op1, Operation* op2) {
	Operation* new_op = new Add();

	new_op->ins.push_back(op1);
	new_op->ins.push_back(op2);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::sub(Operation* op1, Operation* op2) {
	return this->add(op1, this->mul(this->variable(-1), op2));
}

Operation* Graph::mul(Operation* op1, Operation* op2) {
	Operation* new_op = new Mul();

	new_op->ins.push_back(op1);
	new_op->ins.push_back(op2);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::div(Operation* op1, Operation* op2) {
	Operation* new_op = new Div();

	new_op->ins.push_back(op1);
	new_op->ins.push_back(op2);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::log(Operation* op) {
	Operation* new_op = new Log();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::exp(Operation* op) {
	Operation* new_op = new Exp();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::pow(Operation* op1, Operation* op2) {
	return this->exp(this->mul(this->log(op1), op2));
}

Operation* Graph::sqrt(Operation *op) {
	return this->pow(op, this->variable(0.5));
}

Operation* Graph::sin(Operation *op) {
	Operation* new_op = new Sin();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::cos(Operation *op) {
	Operation* new_op = new Cos();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::tan(Operation *op) {
	return this->div(this->sin(op), this->cos(op));
}

Operation* Graph::asin(Operation *op) {
	Operation* new_op = new Asin();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::acos(Operation *op) {
	Operation* new_op = new Acos();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

Operation* Graph::atan(Operation *op) {
	Operation* new_op = new Atan();

	new_op->ins.push_back(op);

	(this->ops).push_back(new_op);

	return new_op;
}

void Graph::zero_grads() {
	for (Operation* op: this->ops) {
		op->grad = 0;
	}
}

double Graph::run(Operation* output) {
	int n_ops = (this->ops).size();

	for (int i = 0; i < n_ops; i++) {
		this->ops[i]->forward();
	}

	this->zero_grads();

	output->grad = 1;

	for (int i = n_ops-1; i >= 0; i--) {
		this->ops[i]->backward();
	}

	return output->result;
}