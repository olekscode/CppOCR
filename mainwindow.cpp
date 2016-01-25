#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    _data_rdr = new MNISTReader("../data/MNIST/t10k-images.idx3-ubyte",
                                "../data/MNIST/t10k-labels.idx1-ubyte");
}

MainWindow::~MainWindow()
{
    delete _data_rdr;
    delete ui;
}

void MainWindow::on_showButton_clicked()
{
    int index = ui->lineEdit->text().toInt();
    QString data = QString(_data_rdr->label(index)) + "\n\n";

    for (int i = 0; i < _data_rdr->rows(); ++i) {
        for (int j = 0; j < _data_rdr->cols(); ++j) {
            data += QString::number(
                        (int)_data_rdr->image(index)[i * _data_rdr->rows() + j]
                    ) + " ";
        }
        data += "\n";
    }

    ui->textBrowser->setPlainText(data);
}
