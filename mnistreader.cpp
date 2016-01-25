#include "mnistreader.h"

MNISTReader::MNISTReader(const char *images_path,
                         const char *labels_path)
{
    _read_images(images_path);
    _read_labels(labels_path);
}

int MNISTReader::_reverse_int(int i) const
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNISTReader::_read_images(const char *path)
{
    std::ifstream file (path, std::ios::binary);
    if (file.is_open()) {
        int magic_number;

        file.read((char*)&magic_number, 4);
        magic_number = _reverse_int(magic_number);

        file.read((char*)&_data_size, 4);
        _data_size = _reverse_int(_data_size);

        file.read((char*)&_n_rows, 4);
        _n_rows = _reverse_int(_n_rows);

        file.read((char*)&_n_cols, 4);
        _n_cols = _reverse_int(_n_cols);

        int img_size = _n_rows * _n_cols;
        _images = new unsigned char*[_data_size];

        for(int i = 0; i < _data_size; ++i) {
            _images[i] = new unsigned char[img_size];

            for(int j = 0; j < img_size; ++j) {
                file.read((char*)&_images[i][j], 1);
            }
        }

        file.close();
    }
    else {
        throw "TODO: Exception";
    }
}

void MNISTReader::_read_labels(const char *path)
{
    std::ifstream file (path, std::ios::binary);
    if (file.is_open()) {
        int magic_number;
        int num_of_labels;

        file.read((char*)&magic_number, 4);
        magic_number = _reverse_int(magic_number);

        file.read((char*)&num_of_labels, 4);
        num_of_labels = _reverse_int(num_of_labels);

        if (num_of_labels != _data_size) {
            throw "TODO: Exception";
        }

        _labels = new unsigned char[_data_size];
        int buf;

        for(int i = 0; i < _data_size; ++i) {
            file.read((char*)&buf, 1);
            _labels[i] = '0' + buf;
        }

        file.close();
    }
    else {
        throw "TODO: Exception";
    }
}
