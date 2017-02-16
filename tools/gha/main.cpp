#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>

#include <af/util.h>
#include <gha.h>

std::size_t input_size = 7;

elephant::gha_solver g(2,input_size,0.01);

int main(int argc, char** argv) {
	if (argc < 2)
		return 1;

	std::vector<float> data_vector;
	std::ifstream data_file(argv[1]);
	std::istream_iterator<double> eos;
	std::istream_iterator<double> it(data_file);

	std::copy(it, eos, std::back_inserter(data_vector));

	af::array data_array(input_size, data_vector.size()/input_size, data_vector.data());

	std::cout << "Hello, world!" << std::endl;
	for (std::size_t epoch = 0; epoch < 100; ++epoch) {
		std::cout << "Epoch " << epoch << std::endl;
		for (std::size_t i = 0; i < data_array.dims(1); ++i) {
			g(data_array.col(i));
		}
	}
	af::print("weight", g.weight());

	return 0;
}
