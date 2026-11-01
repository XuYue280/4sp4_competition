// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "sparse_nn.h"
#include <cmath>
#include <algorithm>

namespace swiftware::hpp {


    // TODO: IMPLEMENT THIS
  DenseMatrix *sparseNNSpmm(DenseMatrix *InData,
                            CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
    int batchSize = InData->m;
    DenseMatrix *pred = new DenseMatrix(batchSize, 1);


    return pred;
  }

    // TODO: IMPLEMENT THIS
    DenseMatrix *sparseNNSpmv(DenseMatrix *InData,
                              CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
        int batchSize = InData->m;
        DenseMatrix *pred = new DenseMatrix(batchSize, 1);


        return pred;
    }


}