#include <gha.h>

#include <af/blas.h>
#include <af/data.h>
#include <af/random.h>
#include <af/algorithm.h>
#include <af/util.h>
#include <af/gfor.h>
#include <af/lapack.h>

#include <iostream>

namespace {

static inline af::array batch_1(const af::array &lhs, const af::array &rhs) {
	return lhs * rhs;
}
static inline af::array batch_2(const af::array &lhs, const af::array &rhs) {
	return lhs - rhs;
}

}

namespace elephant {

gha_solver::gha_solver(std::size_t features, std::size_t input_size, float learning_rate):
	learning_rate_(af::constant(learning_rate, 1, features)),
	output_(),
	weight_(af::randu(input_size, features)*2 - static_cast<float>(1)),
	delta_weight_() {

}

void gha_solver::operator() (const af::array& input) {
	do_operator(input);
}

void gha_solver::operator() (const af::array::array_proxy& input) {
	do_operator(input);
}

template<class T>
inline void gha_solver::do_operator(const T& input) {
	/* dW = nu * [ y_i x_j - y_i Sum [k<=i] w_kj y_k ] */	

	output_ = af::matmul(input, weight_, AF_MAT_TRANS);
	delta_weight_ = af::batchFunc(weight_, output_, &batch_1);
	delta_weight_ = af::accum(delta_weight_, 1);
	delta_weight_ = af::batchFunc(input, delta_weight_, &batch_2);
	delta_weight_ = af::batchFunc(output_, delta_weight_, &batch_1);
	weight_ += af::batchFunc(delta_weight_, learning_rate_, &batch_1);	
}

} // elephant
