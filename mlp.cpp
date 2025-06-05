#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>

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

// Computing the softmax function. 
Vec softmax(const Vec &z){
    size_t o = z.size();
    assert(o > 0);
    double m = z[0]; // storing the max here for stability.

    for (size_t i = 1; i < o; ++i){
        if (z[i] > m){
            m = z[i];
        }
    }
    
    // We're just substracting from the max here for stability reasons.
    // Note that exp(x_{i} - c) / \sum_{j=1}^{n} \exp(x_{j} - c)  
    // = exp(x_{i}) / \sum_{j=1}^{n} \exp(x_{j}) for any constant c > 0.
    double sum = 0.0;   
    for (size_t i = 0; i < o; ++i){
        sum += std::exp(z[i] - m);
    }

    Vec p(o);
    for (size_t i = 0; i < o; ++i){
        p[i] = std::exp(z[i] - m) / sum;
    }
    return p;
}


// Defining a Cache struct to store values.
struct Cache {
    Vec z1; // pre-activation of a hidden layer (d_hidden)
    Vec a1; // activation of hidden layer (d_hidden)
    Vec z2;  // pre-activation of output layer (d_out)
};


// Stores all of the gradients for a 2-layer MLP.
struct Gradients {
    Mat dW1; // size: (d_hidden * d_in)
    Vec db1; // size: (d_hidden)
    Mat dW2; // size: (d_out * d_hidden)
    Vec db2; // size: (d_out)
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

    // Equivalent to z_{2} = W_{2}(ReLU(W_{1}x + b_{1})) + b_{2}. 
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

        return cache.z2;
    }

    void backward(const Vec &x, const Vec &y_true, const Cache &cache, Gradients &grads) const {
        assert(y_true.size() == d_out);
        Vec p = softmax(cache.z2);

        // 1) dL/dz_{2} = p - y_{true}
        Vec dL_dz2(d_out, 0.0);
        for (size_t i = 0; i < d_out; ++i){
            dL_dz2[i] = p[i] - y_true[i];
        }

        // 2) Gradients for W2 and b2. 
        // dW2[i, j] = (dL/dz2[i]) * a1[j]
        grads.dW2.assign(d_out * d_hidden, 0.0);
        for (size_t i = 0; i < d_out; ++i){
            for (size_t j = 0; j < d_hidden; ++j){
                mat_at(grads.dW2, d_out, d_hidden, i, j) = dL_dz2[i] * cache.a1[j];
            }
        }
        // db2[i] = dL/dz2[i]
        grads.db2 = dL_dz2;

        // 3) Backpropagate through z1.
        // dL/da1[j] = \sum_{i=1}^{o} (dL/dz2[i]) * W2[i, j]
        // dL/dz1[j] = dL/da1[j] * 1_{z1[j] > 0}

        Vec dL_dz1(d_hidden, 0.0);
        for (size_t j = 0; j < d_hidden; ++j){
            double grad_sum = 0.0;
            for (size_t i = 0; i < d_out; ++i){
                grad_sum += dL_dz2[i] * mat_at(W2, d_out, d_hidden, i, j);
            }
            double mult = (cache.z1[j] > 0) ? 1.0 : 0.0;
            dL_dz1[j] = grad_sum * mult;
        }

        // 4) Backpropagate through W1 and b1.
        // dW1[i, j] = (dL/dz1[i]) * x[j]
        grads.dW1.assign(d_hidden * d_in, 0.0);
        for (size_t i = 0; i < d_hidden; ++i){
            for (size_t j = 0; j < d_in; ++j){
                mat_at(grads.dW1, d_hidden, d_in, i, j) = dL_dz1[i] * x[j];
            }
        }

        // db1[i] = dL/dz1[i]
        grads.db1 = dL_dz1;
    }
};

// MNIST File Reading.
static uint32_t read_be_uint32(std::ifstream &f){
    unsigned char bytes[4]; 

    // Here we're simply reading in the next 4 bytes, but we first have to do a type cast.
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24)
         | (uint32_t(bytes[1]) << 16)
         | (uint32_t(bytes[2]) << 8)
         |  uint32_t(bytes[3]);
}


int main() {
    // 1) Here we're loading MNIST headers and the associated data.
    std::ifstream ifs_img("data/train-images.idx3-ubyte", std::ios::binary);
    std::ifstream ifs_lbl("data/train-labels.idx1-ubyte", std::ios::binary);
    assert(ifs_img.is_open() && "Cannot open train-images-idx3-ubyte");
    assert(ifs_lbl.is_open() && "Cannot open train-labels-idx1-ubyte");

    // Reading image-file headers.
    uint32_t magic_img = read_be_uint32(ifs_img); 
    uint32_t num_images = read_be_uint32(ifs_img);
    uint32_t num_rows = read_be_uint32(ifs_img);
    uint32_t num_cols = read_be_uint32(ifs_img);

    // Read label-file headers.
    uint32_t magic_lbl = read_be_uint32(ifs_lbl);
    uint32_t num_labels = read_be_uint32(ifs_lbl);

    assert(magic_img == 2051 && "Invalid image-file magic number.");
    assert(magic_lbl == 2049 && "Invalid label-file magic number.");
    assert(num_images == num_labels && "#images != #labels.");

    return 0;
}