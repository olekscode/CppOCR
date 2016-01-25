#ifndef DATAREADER_H
#define DATAREADER_H

// TODO: Find a way to make this class abstract
class DataReader
{
protected:
    // TODO: Try making them constant
    unsigned char** _images;
    unsigned char* _labels;

    int _data_size;
    int _n_rows;
    int _n_cols;

public:
    // We have to ensure that _images and _labels were created
    // (that memory was allocated) before deleting them
    ~DataReader();

    unsigned char* image(int index) const;
    unsigned char label(int index) const;
    int data_size() const;
    int rows() const;
    int cols() const;
};

#endif // DATAREADER_H
