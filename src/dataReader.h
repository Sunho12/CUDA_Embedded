/*
 * =====================================================================================
 *
 *       Filename:  dataReader.h
 *
 *    Description:	For reading data
 *
 *        Version:  1.0
 *        Created:  07/25/2024 12:31:40 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#ifndef __DATAREADER_H__
#define __DATAREADER_H__

#include "predefine.h"

class dataReader{
	public:
	dataReader(const char *train_image_file, const char *train_label_file, const char *test_image_file, const char *test_label_file);
	~dataReader();

	void print_file_info();

	void read_4bytes(FILE *fp, char *input);
	bool read_train_files();
	bool read_test_files();

	void calculate_std_mean();
	void apply_nor_into_trainDB();
	void apply_nor_into_testDB();

	void print_image_and_label(unsigned int idx, bool trainDB = true);

	m_data* get_mnist_db(unsigned idx, bool trainDB = true);

	unsigned get_mnist_db_size(bool trainDB = true);

	private:
	unsigned int m_num_trainDB;
	unsigned int m_num_testDB;
	std::vector<m_data*>	m_trainDB; 
	std::vector<m_data*>	m_testDB;

	float m_std[IMG_SIZE][IMG_SIZE];
	float m_mean[IMG_SIZE][IMG_SIZE];

	unsigned m_dim[2];

	char m_train_image_f[FILENAME_SIZE];
	char m_train_label_f[FILENAME_SIZE];
	char m_test_image_f[FILENAME_SIZE];
	char m_test_label_f[FILENAME_SIZE];
};

#endif
