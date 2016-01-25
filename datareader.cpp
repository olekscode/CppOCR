#include "datareader.h"

DataReader::~DataReader()
{
    for (int i = 0; i < _data_size; ++i) {
        delete _images[i];
    }
    delete _images;
    delete _labels;
}

unsigned char* DataReader::image(int index) const
{
    return _images[index];
}

unsigned char DataReader::label(int index) const
{
    return _labels[index];
}

int DataReader::data_size() const
{
    return _data_size;
}

int DataReader::rows() const
{
    return _n_rows;
}

int DataReader::cols() const
{
    return _n_cols;
}
