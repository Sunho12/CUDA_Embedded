/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Fully-Connected MNIST
 *
 *        Version:  1.0
 *        Created:  07/08/2024 09:54:25 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Sunho Kwak
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */


#include "./src/predefine.h"
#include "./src/dataReader.h"
#include "./src/model.h"
#include "./src/clockMeasure.h"

const char *train_image = "./data/train-images";
const char *train_label = "./data/train-labels";
const char *test_image = "./data/test-images";
const char *test_label = "./data/test-labels";

int main(){
    //Reading train and test images
    dataReader *reader = new dataReader(train_image, train_label, test_image, test_label);

    reader->print_file_info();

    reader->read_train_files();
    reader->read_test_files();
    reader->calculate_std_mean();
    reader->apply_nor_into_trainDB();
    reader->apply_nor_into_testDB();

    //Create FC NN (Read Weight)
    model *fModel = new model(2);
    fModel->read_weights("./layer/Layer01.db");
    fModel->read_weights("./layer/Layer02.db", false);

    // Clock Measure for Comparison
    clockMeasure *ckSingle = new clockMeasure("Single Inference");
    ckSingle->clockReset();
    clockMeasure *ckBatch = new clockMeasure("GPU Batch Inference");
    ckBatch->clockReset();

    const unsigned batch_size = 1000;
    const unsigned tatal_samples = reader->get_mnist_db_size(false);
    const unsigned total_batches = tatal_samples / batch_size;

    // Single Inference
    for (unsigned i = 0; i < tatal_samples; ++i) {
        m_data *img = reader->get_mnist_db(i, false);
        ckSingle->clockResume();
        unsigned result = fModel->perf_forward_exec_on_device(img);
        ckSingle->clockPause();
    }

    // Batch Inference
    for (unsigned b = 0; b < total_batches; ++b){
        m_data batch_data[batch_size];
        for (unsigned i = 0; i < batch_size; ++i) {
            batch_data[i] = *reader->get_mnist_db(b * batch_size + i, false);
        }
        ckBatch->clockResume();
        unsigned char *batch_results = fModel->perf_forward_exec_on_device_batch(batch_data, batch_size);
        free(batch_results);
        ckBatch->clockPause();
    }

    // Print time comparison results
    std::cout << "Performance Comparison:" << std::endl;
    ckSingle->clockPrint();
    ckBatch->clockPrint();

    delete ckSingle;
    delete ckBatch;
    delete reader;
    delete fModel;
    return 0;
}

