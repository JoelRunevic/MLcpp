#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm> 
#include <numeric>
#include <utility>

// Aliases for convenience.
using Vec = std::vector<double>;
using Mat = std::vector<double>; 

// In-place vector/matrix operations for gradient accumulation.
static void add_in_place(Mat &a, const Mat &b) {
    assert(a.size() == b.size()); 
    for (size_t i = 0; i < a.size(); ++i){
        a[i] += b[i];
    }
}

static void scale_in_place(Mat &a, double s) {
    for (size_t i = 0; i < a.size(); ++i){
        a[i] *= s;
    }
}

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

static void sgd_update(std::vector<double> &param, 
                       const std::vector<double> &grad, 
                       double lr) {
    assert(param.size() == grad.size());
    for (size_t i = 0; i < param.size(); ++i){
        param[i] -= lr * grad[i];
    }
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

void init_he_normal(MLP &model, std::mt19937 &rng){
    double std1 = std::sqrt(2.0 / double(model.d_in));
    std::normal_distribution<double> dist1(0.0, std1);
    for (auto &w : model.W1) {
        w = dist1(rng);
    }

    double std2 = std::sqrt(2.0 / double(model.d_hidden));
    std::normal_distribution<double> dist2(0.0, std2);
    for (auto &w : model.W2) {
        w = dist2(rng);
    }
}

// Simple Dataset container.
struct Dataset {
    std::vector<Vec> images; // Each normalized to [0, 1].
    std::vector<int> labels;
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

void load_mnist(const std::string &img_path, const std::string &lbl_path, Dataset &ds){
    std::ifstream ifs_img(img_path, std::ios::binary);
    std::ifstream ifs_lbl(lbl_path, std::ios::binary);
    assert(ifs_img.is_open() && "Cannot open image file");
    assert(ifs_lbl.is_open() && "Cannot open label file");

    // Reading the header information here. 
    uint32_t magic_img = read_be_uint32(ifs_img);
    uint32_t n_images = read_be_uint32(ifs_img);
    uint32_t rows = read_be_uint32(ifs_img);
    uint32_t cols = read_be_uint32(ifs_img);

    uint32_t magic_lbl = read_be_uint32(ifs_lbl);
    uint32_t n_labels = read_be_uint32(ifs_lbl);

    assert(magic_img == 2051 && "Invalid image magic");
    assert(magic_lbl == 2049 && "Invalid label magic");
    assert(n_images == n_labels && "Mismatched image/label counts");

    size_t image_size = size_t(rows) * size_t(cols);

    // Allocating the corresponding memory.
    ds.images.assign(n_images, Vec(image_size));
    ds.labels.assign(n_labels, 0);

    // Read all images (one byte per pixel) and normalize to [0, 1].
    std::vector<unsigned char> buffer(image_size); 
    for (size_t i = 0; i < n_images; ++i){
        ifs_img.read(reinterpret_cast<char*>(buffer.data()), image_size);
        for (size_t j = 0; j < image_size; ++j){
            ds.images[i][j] = buffer[j] / 255.0;
        }
    }

    // Read all labels (one byte each).
    for (size_t i = 0; i < n_labels; ++i){
        unsigned char lb;
        ifs_lbl.read(reinterpret_cast<char*>(&lb), 1);
        ds.labels[i] = static_cast<int>(lb);
    }

}

void compute_metrics(MLP &model, Dataset &ds, Cache &cache, size_t &idx, int &samples, int &correct, double &loss){
    // Getting the logits and associated probabilities.
    Vec logits = model.forward(ds.images[idx], cache);
    Vec probs = softmax(logits);
    
    // Getting argmax (for accuracy) and computing cross-entropy loss.
    int max_idx = 0; 
    for (int j = 0; j < static_cast<int>(logits.size()); ++j){
        if (logits[j] > logits[max_idx]){
            max_idx = j; 
        }
    }

    ++samples;
    if (max_idx == ds.labels[idx]){
        ++correct;
    }
    loss += -std::log(probs[ds.labels[idx]] + 1e-12);
}




int main() {
    // 1) We first load data.
    Dataset train_ds, test_ds; 
    load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", train_ds);
    load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", test_ds);
    
    // 2) Constructing the model itself.
    size_t input_dim = train_ds.images[0].size();
    size_t hidden_dim = 128; 
    size_t output_dim = 10;
    std::mt19937 rng(42); 

    MLP model(input_dim, hidden_dim, output_dim);
    init_he_normal(model, rng);

    // 3) Defining the training settings.
    size_t epochs = 3;
    size_t batch_size = 64; 
    size_t N = train_ds.images.size();
    std::vector<size_t> idx(N); 
    double lr = 0.01;

    // 4) Training loop. 
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        // a) Here, we first shuffle the indices.
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        double running_loss = 0.0; 
        int running_correct = 0; 
        int running_samples = 0;

        // b) Iterate over the batch.
        for (size_t start = 0; start < N; start += batch_size) {
            size_t end = std::min(start + batch_size, N);
            size_t bs = end - start; 

            // Zero out the accumulated gradients.
            Gradients acc;
            acc.dW1.assign(model.d_hidden * model.d_in, 0.0);
            acc.db1.assign(model.d_hidden, 0.0);
            acc.dW2.assign(model.d_out * model.d_hidden, 0.0);
            acc.db2.assign(model.d_out, 0.0);

            Cache cache; 
            Gradients grads;

            // Accumulating per-example gradients.
            for (size_t ii = start; ii < end; ++ii) { 
                size_t i = idx[ii];
                
                compute_metrics(
                    model,
                    train_ds,
                    cache,
                    i,
                    running_samples,
                    running_correct,
                    running_loss
                );
                
                Vec y_true(output_dim, 0.0);
                y_true[train_ds.labels[i]] = 1.0; 

                model.backward(train_ds.images[i], y_true, cache, grads);

                add_in_place(acc.dW1, grads.dW1);
                add_in_place(acc.db1, grads.db1); 
                add_in_place(acc.dW2, grads.dW2);
                add_in_place(acc.db2, grads.db2);
            }

            std::cout << "Start: " << start;
            std::cout << " Loss: " << running_loss / running_samples;
            std::cout << " Accuracy: " << double(running_correct) / running_samples << std::endl;

            // Average the accumalation gradients.
            double inv_bs = 1.0 / static_cast<double>(bs);
            scale_in_place(acc.dW1, inv_bs);
            scale_in_place(acc.db1, inv_bs);
            scale_in_place(acc.dW2, inv_bs);
            scale_in_place(acc.db2, inv_bs);

            sgd_update(model.W1, acc.dW1, lr);
            sgd_update(model.b1, acc.db1, lr);
            sgd_update(model.W2, acc.dW2, lr);
            sgd_update(model.b2, acc.db2, lr);
        }
        std::cout << "Epoch " << epoch << " Completed.";
    }

    // Computing the test set metrics.
    double test_loss = 0.0; 
    int test_samples = 0; 
    int test_correct = 0;
    Cache cache_tmp;

    for (size_t i = 0; i < test_ds.images.size(); ++i){
        compute_metrics(
            model, 
            test_ds,
            cache_tmp,
            i,
            test_samples,
            test_correct,
            test_loss
        );
    }
    std::cout << "\n\n################## TEST SET SUMMARY ##################\n\n";
    std::cout << " Test Loss: " << test_loss / test_samples;
    std::cout << " Accuracy: " << double(test_correct) / test_samples << std::endl;

    return 0;
}