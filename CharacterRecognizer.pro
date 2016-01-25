#-------------------------------------------------
#
# Project created by QtCreator 2016-01-23T05:14:19
#
#-------------------------------------------------

QT       += core gui
CONFIG   += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CharacterRecognizer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    mnistreader.cpp \
    datareader.cpp

HEADERS  += mainwindow.h \
    mnistreader.h \
    datareader.h

FORMS    += mainwindow.ui
