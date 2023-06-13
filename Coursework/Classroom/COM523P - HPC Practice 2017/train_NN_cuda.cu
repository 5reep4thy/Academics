%%cu
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include<sys/time.h>
using namespace std;

// Training image file name
const string training_image_fn = "/content/drive/My Drive/Cuda_project_test/MNIST/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "/content/drive/My Drive/Cuda_project_test/MNIST/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "training-report.dat";

// Number of training samples
const int nTraining = 1;

// Image size in MNIST database
const int width = 28;
const int height = 28;

const int blockSize = 512, gridSize = 1;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 100000; 
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 1;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;
///////////////////////////////////////////////////////////
__managed__ double part_matmul[blockSize][n3 + 1];
// From layer 1 to layer 2. Or: Input layer - Hidden layer
__managed__ double w1[n1 + 1][n2 + 1], delta1[n1 + 1][n2 + 1], out1[n1 + 1];
__managed__ int b[10][10];
// From layer 2 to layer 3. Or; Hidden layer - Output layer
__managed__ double w2[n2 + 1][n3 + 1], delta2[n2 + 1][n3 + 1], in2[n2 + 1], out2[n2 + 1], theta2[n2 + 1];

// Layer 3 - Output layer
__managed__ double in3[n3 + 1], out3[n3 + 1], theta3[n3 + 1];
__managed__ double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
__managed__ int d[width + 1][height + 1];

double t1 = 0, t2 = 0, total_time = 0;
///////////////////////////////////////////////////////////
double cpuSecond() 
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//////////////////////////////////////////////////////////////
// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
	cout << "Training label data: " << training_label_fn << endl;
	cout << "No. training sample: " << nTraining << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

__global__ void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int bs = blockDim.x;
    
    // Initialization for weights from Input layer to Hidden layer
    for (int j = id + 1; j <= n2; j += bs) {
         for (int i = 1; i <= n1; ++i) {
            w1[i][j] = 0;
        }
	}
	
	// Initialization for weights from Hidden layer to Output layer
    for (int i = id + 1; i <= n2; i += bs) {
        for (int j = 1; j <= n3; ++j) {
            w2[i][j] = 0;
        }
	}
}

// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+

__global__ void perceptron() {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int bs = blockDim.x;
    for (int i = id + 1; i <= n2; i += bs) {
		in2[i] = 0.0;
	}
    //printf("Testing xxxxx %d, %d \n", id, bs);
    __syncthreads();
    for (int i = id + 1; i <= n3; i += bs) {
		in3[i] = 0.0;
	}
    __syncthreads();
    for (int i = 1; i <= n1; ++i) {
        for (int j = id + 1; j <= n2; j += bs) {
            in2[j] += out1[i] * w1[i][j];
		}
	}
    __syncthreads();
    for (int i = id + 1; i <= n2; i += bs) {
		out2[i] = 1.0 / (1.0 + exp(float(-in2[i])));
	}

    __syncthreads();
    if (id == 0) {
        for (int i = 1; i <= n3; i++) {
            for (int j = 0; j < bs; j++) {
                part_matmul[j][i] = 0;
            }
        }
    }
    __syncthreads();
    for (int i = id + 1; i <= n2; i += bs) {
            for (int j = 1; j <= n3; ++j)
            {
                part_matmul[id][j] += out2[i] * w2[i][j];
            }
    }
    
    __syncthreads();
    if (id == 0) {
        for (int i = 1; i <= n3; i++) {
            for (int j = 0; j < bs; j++) {
                in3[i] += part_matmul[j][i];
            }
        }
    }

    __syncthreads();
    /*
    if (id == 0) {
        for (int i = 1; i <= n2; ++i) {
            for (int j = 1; j <= n3; ++j)
            {
                in3[j] += out2[i] * w2[i][j];
            }
        }
    }
    */
    __syncthreads();
    if (id == 0) {
        for (int i = 1; i <= n3; ++i) {
            out3[i] = 1.0 / (1.0 + exp(float(-in3[i])));
        }
    }
    __syncthreads();
}


// +---------------+
// | Norm L2 error |
// +---------------+

double square_error(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+


__global__ void back_propagation() {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int bs = blockDim.x;
    double sum;
    __syncthreads();
    if (id == 0) {
        for (int i = 1; i <= n3; ++i) {
            theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
        }
    }
    __syncthreads();
    
    for (int i = id + 1; i <= n2; i += bs) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }
    __syncthreads();
    
    for (int i = id + 1; i <= n2; i += bs) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
    }
    __syncthreads();
    
    for (int j = id + 1 ; j <= n2 ; j += bs )  {
        for (int i = 1; i <= n1; ++i) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
    }

    __syncthreads();
}



// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+

int learning_process() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			delta1[i][j] = 0.0;
		}
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			delta2[i][j] = 0.0;
		}
	}
    total_time = 0;
    for (int i = 1; i <= epochs; ++i) {

        t1 = cpuSecond();
        perceptron<<<blockSize, gridSize>>>();
        
        cudaDeviceSynchronize();
        back_propagation<<<blockSize, gridSize>>>();
        
        t2 = cpuSecond();
        total_time += t2 - t1;
        cout << square_error() << " after " << i << " iterations\n";
        /*
        if (square_error() < epsilon) {
			return i;
		}
        */
    }
    cout << "Total time elapsed = " << total_time << "\n";
    return epochs;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

void input() {
	// Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}
	
	cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << d[i][j];
		}
		cout << endl;
	}

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}

	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
    
    cout << "Label: " << (int)(number) << endl;
}

// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file << w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}


// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
	about();
	
    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file
  
    
	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Neural Network Initialization
    init_array<<<blockSize, gridSize>>>();
    cudaDeviceSynchronize();
    for (int sample = 1; sample <= nTraining; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        input();
		
		// Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning_process();
        
		// Write down the squared error
		cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
		
		// Save the current network (weights)
		if (sample % 100 == 0) {
			cout << "Saving the network to " << model_fn << " file." << endl;
			write_matrix(model_fn);
		}
    }
	
	// Save the final network
    write_matrix(model_fn);

    report.close();
    image.close();
    label.close();
    
    return 0;
}
