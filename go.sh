#!/bin/bash
cd  build
cmake ..
make -j2 
sudo make install 

sudo cp /usr/local/etc/OpenCL/vendors/pocl.icd  /etc/OpenCL/vendors/
