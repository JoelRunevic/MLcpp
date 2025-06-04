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

// Defining a Cache struct to store values.
struct Cache {
    Vec z1; // pre-activation of a hidden layer (d_hidden)
    Vec a1; // activation of hidden layer (d_hidden)
    Vec z2;  // pre-activation of output layer (d_out)
};


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

    // Just applying an affine transformation: z = Ax + b. 
    // For now, this is hard-coded to two layers and does not have much modularity.
    Vec forward(const Vec &x, Cache &cache) const {
        assert(x.size() == d_in);

        // 1) z_{1} = W_{1}x + b_{1}
        cache.z1.assign(d_hidden, 0.0);
        for (size_t i = 0; i < d_hidden; ++i){
            double sum = 0.0;
            for (size_t j = 0; j < d_in; ++j){
                sum += mat_at(W1, d_hidden, d_in, i, j) * x[j];
            }
            cache.z1[i] = sum + b1[i];
        }

        // 2) a_{1} = ReLU(z_{1})
        cache.a1 = cache.z1; 
        for (size_t i = 0; i < d_hidden; ++i){
            if (cache.a1[i] < 0){
                cache.a1[i] = 0;
            }
        }
        
        // 3) z_{2} = W_{2}a_{1} + b_{2}
        cache.z2.assign(d_out, 0.0);
        for (size_t i = 0; i < d_out; ++i){
            double sum = 0.0; 
            for (size_t j = 0; j < d_hidden; ++j){
                sum += mat_at(W2, d_out, d_hidden, i, j) * cache.a1[j];
            }
            cache.z2[i] = sum + b2[i];
        }

        // Equivalent to z_{2} = W_{2}(ReLU(W_{1}x + b_{1})) + b_{2}
        return cache.z2;
    }

};


int main() {
    // Doing a toy example for correctness.
    MLP net(2, 3, 1);
    Cache cache;

    for (double &w : net.W1){
        w = 1.0;
    }
    std::fill(net.b1.begin(), net.b1.end(), 2.0);

    for (double &w : net.W2){
        w = 1.0; 
    }
    std::fill(net.b2.begin(), net.b2.end(), 2.0);

    Vec x = {2.0, 5.0};
    Vec z2 = net.forward(x, cache);

    std::cout << "z_{1} = ";
    for (double v : cache.z1) std::cout << v << " ";

    std::cout << "\na_{1} = "; 
    for (double v : cache.a1) std::cout << v << " ";

    std::cout << "\nz_{2} = ";
    for (double v : cache.z2) std::cout << v << " ";

    return 0;
}