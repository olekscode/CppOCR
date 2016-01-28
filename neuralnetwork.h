#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "datareader.h"

// TEMPORARY
#include <QDebug>
#include <QString>

// TODO: Consider creating a separate Data class
//       So that DataReader returned Data

class NeuralNetwork
{
    // Including the input and output layers
    int _layers;
    // Excluding the biases
    int* _layer_sizes;      // [_layers]

    double* _biases;        // [_layers-1]
    double (*_activ_func)(double);
    double (*_der_activ_func)(double);

    double*** _thetas;      // [_layers-1][_layer_sizes[k]+1][_layer_sizes[k+1]]

    // These should be created when train() is called
    // And deleted at the end of training
    double** _activations;  // [_layers-1][_layer_sizes[k+1]]
    double** _z;            // [_layers-1][_layer_sizes[k+1]]
    double** _deltas;       // [_layers-1][_layer_sizes[k+1]+1]
    double*** _cap_deltas;  // [_layers-1][_layer_sizes[k]+1][_layer_sizes[k+1]]

    // TODO: Consider using static arrays
    // QUESTION: When they are better?
    // int _layer_sizes[];

public:
    NeuralNetwork(int layers,
                  int* layer_sizes,
                  double* biases,
                  double (*activ_func)(double),
                  double (*der_activ_func)(double));

    NeuralNetwork(const NeuralNetwork& other);
    ~NeuralNetwork();

    NeuralNetwork& operator= (const NeuralNetwork& other);

    int layers() const;
    int layer_size(int index) const;
    double*** thetas() const;

    // Because it's a bit confusing to pass a DataReader
    // instead of Data (while in fact it's a collection
    // of data)
    void train(DataReader* data);
    char predict(unsigned char* img) const;

private:
    // TODO: Pass this function as a constructor parameter.
    //       It is one of the NN settings
    void _set_init_theta(DataReader* data);

    void _feedforward(unsigned char* img);
    void _backpropagate(unsigned char* img, char label);
    void _regularize();

    // TEMPORARY
    int __y_bin(char c, int index) const;
    char __to_char(double* out) const;
};

#endif // NEURALNETWORK_H
