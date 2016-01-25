#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <fstream>

#include "datareader.h"

// INSPIRED BY: https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
class MNISTReader : public DataReader
{
public:
    MNISTReader(const char* images_path,
                const char* labels_path);

private:
    void _read_images(const char* path);
    void _read_labels(const char* path);
    int _reverse_int(int i) const;
};

#endif // MNISTREADER_H
