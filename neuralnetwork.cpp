#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(int layers,
                             int* layer_sizes,
                             double* biases,
                             double (*activ_func)(double),
                             double (*der_activ_func)(double))
    : _layers(layers),
      _activ_func(activ_func),
      _der_activ_func(der_activ_func)
{
    _layer_sizes = new int[_layers];

    for (int i = 0; i < _layers; ++i) {
        _layer_sizes[i] = layer_sizes[i];
    }

    _biases = new double[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        _biases[i] = biases[i];
    }

    _thetas = new double**[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        // +1 bias unit
        _thetas[i] = new double*[_layer_sizes[i] + 1];

        for (int j = 0; j < _layer_sizes[i] + 1; ++j) {
            _thetas[i][j] = new double[_layer_sizes[i + 1]];
        }
    }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
    : _layers(other._layers),
      _activ_func(other._activ_func),
      _der_activ_func(other._der_activ_func)
{
    _layer_sizes = new int[_layers];

    for (int i = 0; i < _layers; ++i) {
        _layer_sizes[i] = other._layer_sizes[i];
    }

    _biases = new double[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        _biases[i] = other._biases[i];
    }

    _thetas = new double**[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        // +1 bias unit
        _thetas[i] = new double*[_layer_sizes[i] + 1];

        for (int j = 0; j < _layer_sizes[i] + 1; ++j) {
            _thetas[i][j] = new double[_layer_sizes[i + 1]];

            for (int k = 0; k < _layer_sizes[i + 1]; ++k) {
                _thetas[i][j][k] = other._thetas[i][j][k];
            }
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{
    for (int i = 0; i < _layers - 1; ++i) {
        for (int j = 0; j < _layer_sizes[i] + 1; ++j) {
            delete[] _thetas[i][j];
        }
        delete[] _thetas[i];
    }
    delete[] _thetas;
    delete[] _biases;
    delete[] _layer_sizes;
}

NeuralNetwork& NeuralNetwork::operator= (const NeuralNetwork& other)
{
    // TODO: Find out if there is a way to avoid copying
    //       all this code from copy constructor
    _activ_func = other._activ_func;
    _der_activ_func = other._der_activ_func;

    _layers = other._layers;
    _layer_sizes = new int[_layers];

    for (int i = 0; i < _layers; ++i) {
        _layer_sizes[i] = other._layer_sizes[i];
    }

    _biases = new double[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        _biases[i] = other._biases[i];
    }

    _thetas = new double**[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        // +1 bias unit
        _thetas[i] = new double*[_layer_sizes[i] + 1];

        for (int j = 0; j < _layer_sizes[i] + 1; ++j) {
            _thetas[i][j] = new double[_layer_sizes[i + 1]];

            for (int k = 0; k < _layer_sizes[i + 1]; ++k) {
                _thetas[i][j][k] = other._thetas[i][j][k];
            }
        }
    }
    return *this;
}

int NeuralNetwork::layers() const
{
    return _layers;
}

int NeuralNetwork::layer_size(int index) const
{
    return _layer_sizes[index];
}

double*** NeuralNetwork::thetas() const
{
    return _thetas;
}

void NeuralNetwork::train(DataReader* data)
{
    if (data->rows() * data->cols() != _layer_sizes[0]) {
        throw "Error";
    }

    _set_init_theta(data);

    // First row of activation units is equal to the
    // traing example (input). Therefore there is no
    // need to allocate more memory and take time to
    // copy this whole layer
    _activations    = new double*[_layers - 1];
    _z              = new double*[_layers - 1];
    _deltas         = new double*[_layers - 1];
    _cap_deltas     = new double**[_layers - 1];

    for (int i = 0; i < _layers - 1; ++i) {
        // Excluding biases
        _activations[i] = new double[_layer_sizes[i + 1]];
        _z[i]           = new double[_layer_sizes[i + 1]];
        _deltas[i]      = new double[_layer_sizes[i + 1] + 1];
        _cap_deltas[i]  = new double*[_layer_sizes[i] + 1];

        for (int j = 0; j < _layer_sizes[i] + 1; ++j) {
            _cap_deltas[i][j] = new double[_layer_sizes[i + 1]];
        }
    }

    for (int k = 0; k < _layers - 1; ++k) {
        for (int i = 0; i < _layer_sizes[k] + 1; ++i) {
            for (int j = 0; j < _layer_sizes[k + 1]; ++j) {
                _cap_deltas[k][i][j] = 0;
            }
        }
    }

//    for (int i = 0; i < data->data_size(); ++i) {
    for (int i = 0; i < 1; ++i) {
        _feedforward(data->image(i));
        _backpropagate(data->image(i), data->label(i));
    }

    for (int i = 0; i < _layers - 1; ++i) {
        for (int j = 0; j < _layer_sizes[i] + 1; ++j) {
            delete[] _cap_deltas[i][j];
        }
        delete[] _cap_deltas[i];

        delete[] _activations[i];
        delete[] _z[i];
        delete[] _deltas[i];
    }
    delete[] _activations;
    delete[] _z;
    delete[] _deltas;
    delete[] _cap_deltas;

    _regularize();
}

char NeuralNetwork::predict(unsigned char* img) const
{
    double* activ = new double[_layer_sizes[1]];
    double* activ_prev;

    for (int i = 0; i < _layer_sizes[1]; ++i) {
        activ[i] = _biases[0] * _thetas[0][0][i];

        for (int j = 0; j < _layer_sizes[0]; ++j) {
            activ[i] += img[j] * _thetas[0][j + 1][i];
        }
        activ[i] = _activ_func(activ[i]);
    }

    for (int k = 1; k < _layers - 1; ++k) {
        activ_prev = activ;
        activ = new double[_layer_sizes[k + 1]];

        for (int i = 0; i < _layer_sizes[k + 1]; ++i) {
            activ[i] = _biases[k] * _thetas[k][0][i];

            for (int j = 0; j < _layer_sizes[k]; ++j) {
                activ[i] += activ_prev[j] * _thetas[k][j + 1][i];
            }
            activ[i] = _activ_func(activ[i]);
        }
        delete[] activ_prev;
    }

    QString out = QString();
    for (int i = 0; i < _layer_sizes[_layers - 1]; ++i) {
        out += QString::number(activ[i]) + " ";
    }
    qDebug() << out;

    char res = __to_char(activ);
    delete[] activ;

    return res;
}

void NeuralNetwork::_set_init_theta(DataReader* data)
{
    for (int k = 0; k < _layers - 1; ++k) {
        for (int i = 0; i < _layer_sizes[k] + 1; ++i) {
            for (int j = 0; j < _layer_sizes[k + 1]; ++j) {
                _thetas[k][i][j] = 1;
            }
        }
    }
}

void NeuralNetwork::_feedforward(unsigned char* img)
{
    // TODO: Fix the mess with i & j
    for (int i = 0; i < _layer_sizes[1]; ++i) {
        _z[0][i] = _biases[0] * _thetas[0][0][i];

        for (int j = 0; j < _layer_sizes[0]; ++j) {
            _z[0][i] += img[j] * _thetas[0][j + 1][i];
        }
        _activations[0][i] = _activ_func(_z[0][i]);
    }

    for (int k = 1; k < _layers - 1; ++k) {
        for (int i = 0; i < _layer_sizes[k + 1]; ++i) {
            _z[k][i] = _biases[k] * _thetas[k][0][i];

            for (int j = 0; j < _layer_sizes[k]; ++j) {
                _z[k][i] += _activations[k - 1][j] * _thetas[k][j + 1][i];
            }
            _activations[k][i] = _activ_func(_z[k][i]);
        }
    }
}

void NeuralNetwork::_backpropagate(unsigned char* img, char label)
{
    // _deltas[_layers - 2] and _activations[_layers - 2]
    // are actually the last elements of these arrays
    for (int i = 0; i < _layer_sizes[_layers - 1]; ++i) {
        _deltas[_layers - 2][i] = _activations[_layers - 2][i] - __y_bin(label, i);
    }

    for (int k = _layers - 3; k > 0; --k) {
        for (int i = 0; i < _layer_sizes[k + 1] + 1; ++i) {
            _deltas[k][i] = 0;

            for (int j = 0; j < _layer_sizes[k + 2]; ++j) {
                _deltas[k][i] += _thetas[k + 1][i][j] * _deltas[k + 1][j];
            }
            _deltas[k][i] *= _der_activ_func(_z[k][i]);
        }
    }

    for (int k = 0; k < _layers - 1; ++k) {
        for (int j = 0; j < _layer_sizes[k + 1]; ++j) {
            // WARNING: Potential logical error due to j < _layer_sizes[k + 1]
            _cap_deltas[k][0][j] += _biases[k] * _deltas[k][0];
        }
    }

    for (int i = 1; i < _layer_sizes[0] + 1; ++i) {
        for (int j = 0; j < _layer_sizes[1]; ++j) {
            // WARNING: Potential logical error at _deltas[0][j + 1]
            _cap_deltas[0][i][j] += img[i - 1] * _deltas[0][j + 1];
        }
    }

    for (int k = 1; k < _layers - 1; ++k) {
        for (int i = 1; i < _layer_sizes[k] + 1; ++i) {
            for (int j = 0; j < _layer_sizes[k + 1]; ++j) {
                // WARNING: Potential logical error at _deltas[k][j + 1]
                _cap_deltas[k][i][j] += _activations[k - 1][j] * _deltas[k][j + 1];
            }
        }
    }
}

void NeuralNetwork::_regularize()
{

}

// =======================================================
int NeuralNetwork::__y_bin(char c, int index) const
{
    if ('0' + index == c) {
        return 1;
    }
    else {
        return 0;
    }
}

char NeuralNetwork::__to_char(double *out) const
{
    int index_of_max = 0;

    for (int i = 1; i < _layer_sizes[_layers - 1]; ++i) {
        if (out[i] > out[index_of_max]) {
            index_of_max = i;
        }
    }
    return '0' + index_of_max;
}
