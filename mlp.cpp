#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>

// Aliases for convenience.
using Vec = std::vector<double>;
using Mat = std::vector<double>; 

// Helper function in order to index into a row-major matrix.
inline double& mat_at(Mat &m, size_t rows, size_t cols, size_t i, size_t j){
    assert(i < rows && j < cols);
    return m[i * cols + j]; 
}

inline double mat_at(const Mat &m, size_t rows, size_t cols, size_t i, size_t j){
    assert(i < rows && j < cols);
    return m[i * cols + j]; 
}


// Defining the intial MLP struct.
struct MLP {
    size_t d_in; 
    size_t d_hidden; 
    size_t d_out; 

    // Weight matrices and bias vectors. Will refactor this later.
    Mat W1; // size: (d_hidden x d_in)
    Vec b1; // size: (d_hidden)

    Mat W2; // size: (d_out x d_hidden)
    Vec b2; // size: (d_out)

    // Constructor: allocating storage.
    MLP(size_t input_dim, size_t hidden_dim, size_t output_dim)
        : d_in(input_dim), 
          d_hidden(hidden_dim), 
          d_out(output_dim), 
          W1(hidden_dim * input_dim, 0.0),
          b1(hidden_dim, 0.0), 
          W2(output_dim * hidden_dim, 0.0),
          b2(output_dim, 0.0)
    {
        assert(d_in > 0 && d_hidden > 0 && d_out > 0);
    }
};


int main() {
    return 0; 
}