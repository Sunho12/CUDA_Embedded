/*
 * =====================================================================================
 *
 *       Filename:  dataReader.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/25/2024 09:16:38 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#include "dataReader.h"


dataReader::dataReader(const char *train_image_file, const char *train_label_file, const char *test_image_file, const char *test_label_file){
		strcpy(m_train_image_f, train_image_file);
		strcpy(m_train_label_f, train_label_file);
		strcpy(m_test_image_f, test_image_file);
		strcpy(m_test_label_f, test_label_file);

		m_num_trainDB = 0;
		m_num_testDB = 0;

		m_dim[0] = 0;
		m_dim[1] = 0;

		for(int y = 0; y < IMG_SIZE; y++){
			for(int x = 0; x < IMG_SIZE; x++){
				m_std[y][x] = 0.0f;
				m_mean[y][x] = 0.0f;
			}
		}
}

dataReader::~dataReader(){
	for(auto iter = m_trainDB.begin(); iter != m_trainDB.end(); iter++){
		free(*iter);
	}
	for(auto iter = m_testDB.begin(); iter != m_testDB.end(); iter++){
		free(*iter);
	}
	m_trainDB.clear();
	m_testDB.clear();
}

void dataReader::print_file_info(){
	printf("[BEGIN] showing file info:\n");
	printf("\t-Train Image File: %s\n", m_train_image_f);
	printf("\t-Train Label File: %s\n", m_train_label_f);
	printf("\t-Test Image File: %s\n", m_test_image_f);
	printf("\t-Test Label File: %s\n", m_test_label_f);
	printf("[END] showing file info\n");
}

void dataReader::read_4bytes(FILE *fp, char *input){
	for(int i = 0; i < 4; i++){
		fscanf(fp, "%c", &input[3-i]);
	}
}

void dataReader::print_image_and_label(unsigned int idx, bool trainDB){
	m_data *tmp = nullptr;
	if(trainDB){
		if(idx < m_trainDB.size()){
			tmp = m_trainDB[idx];
		}
	}else{
		if(idx < m_testDB.size()){
			tmp = m_testDB[idx];
		}
	}

	if(!tmp){
		printf("[ERROR] IDX is larger than size of DB\n");
	}else{
		printf("Label: %u\n", (unsigned int)tmp->label);
		for(int i = 0; i < m_dim[0]; i++){
			for(int j = 0; j < m_dim[1]; j++){
				if(tmp->data[i][j] == 0){
					printf(" ");
				}else{
					printf("*");
				}
			}
			printf("\n");
		}

	}
}

bool dataReader::read_train_files(){
	bool ret = true;

	FILE *i_fp = fopen(this->m_train_image_f, "r");
	FILE *l_fp = fopen(this->m_train_label_f, "r");
	
	if(i_fp == nullptr && l_fp == nullptr){
		ret = false;
		goto cleanup;
	}
	
	c2u tmp;
	
	read_4bytes(i_fp, tmp.c_array);
	if(tmp.uint != 2051){
		ret = false;
		printf("[ERROR] Train Image File is not valid\n");
		goto cleanup;
	}

	read_4bytes(i_fp, tmp.c_array);
	m_num_trainDB = tmp.uint;

	read_4bytes(l_fp, tmp.c_array);
	if(tmp.uint != 2049){
		ret = false;
		printf("[ERROR] Train Label File is not valid\n");
		goto cleanup;
	}

	read_4bytes(l_fp, tmp.c_array);
	
	if(m_num_trainDB != tmp.uint){
		ret = false;
		printf("[ERROR] Train Image and Label Counts are different\n");
		goto cleanup;
	}
	
	for(int i = 0; i < 2; i++){
		read_4bytes(i_fp, tmp.c_array);
		m_dim[i] = tmp.uint;
	}
	
	assert(m_dim[0] == m_dim[1]);
	assert(m_dim[0] ==  IMG_SIZE);

	for(int i = 0; i < m_num_trainDB; i++){
		m_data *tmpDB = (m_data*)malloc(sizeof(m_data));
		fread(tmpDB->data, 1, m_dim[0] * m_dim[1], i_fp);
		fread(&tmpDB->label, 1, 1, l_fp);
		m_trainDB.push_back(tmpDB);
	}

	assert(m_num_trainDB == m_trainDB.size());
	printf("[Train DB] %lu images are loaded\n", m_trainDB.size());

cleanup:
	fclose(i_fp);
	fclose(l_fp);

	return true;
}

bool dataReader::read_test_files(){
	bool ret = true;

	FILE *i_fp = fopen(this->m_test_image_f, "r");
	FILE *l_fp = fopen(this->m_test_label_f, "r");
	
	if(i_fp == nullptr && l_fp == nullptr){
		ret = false;
		goto cleanup;
	}
	
	c2u tmp;
	
	read_4bytes(i_fp, tmp.c_array);
	if(tmp.uint != 2051){
		ret = false;
		printf("[ERROR] Test Image File is not valid\n");
		goto cleanup;
	}

	read_4bytes(i_fp, tmp.c_array);
	m_num_testDB = tmp.uint;

	read_4bytes(l_fp, tmp.c_array);
	if(tmp.uint != 2049){
		ret = false;
		printf("[ERROR] Test Label File is not valid\n");
		goto cleanup;
	}

	read_4bytes(l_fp, tmp.c_array);
	
	if(m_num_testDB != tmp.uint){
		ret = false;
		printf("[ERROR] Test Image and Label Counts are different\n");
		goto cleanup;
	}
	
	for(int i = 0; i < 2; i++){
		read_4bytes(i_fp, tmp.c_array);
		m_dim[i] = tmp.uint;
	}
	
	assert(m_dim[0] == m_dim[1]);
	assert(m_dim[0] ==  IMG_SIZE);

	for(int i = 0; i < m_num_testDB; i++){
		m_data *tmpDB = (m_data*)malloc(sizeof(m_data));
		fread(tmpDB->data, 1, m_dim[0] * m_dim[1], i_fp);
		fread(&tmpDB->label, 1, 1, l_fp);
		m_testDB.push_back(tmpDB);
	}

	assert(m_num_testDB == m_testDB.size());
	printf("[Test DB] %lu images are loaded\n", m_testDB.size());

cleanup:
	fclose(i_fp);
	fclose(l_fp);

	return true;
}

void dataReader::calculate_std_mean(){
	printf("[STD MEAN] Calculating Std and Mean\n");
	assert(m_trainDB.size() > 0);
	
	for(auto iter = m_trainDB.begin(); iter != m_trainDB.end(); iter++){
		for(int y = 0; y < IMG_SIZE; y++){
			for(int x = 0; x < IMG_SIZE; x++){
				m_mean[y][x] += (float)(*iter)->data[y][x];
			}
		}
	}
	for(int y = 0; y < IMG_SIZE; y++){
		for(int x = 0; x < IMG_SIZE; x++){
			m_mean[y][x] /= (float)m_trainDB.size();
		}
	}

	for(auto iter = m_trainDB.begin(); iter != m_trainDB.end(); iter++){
		for(int y = 0; y < IMG_SIZE; y++){
			for(int x = 0; x < IMG_SIZE; x++){
				m_std[y][x] += pow((float)(*iter)->data[y][x] - m_mean[y][x], 2.0f);
			}
		}
	}
	for(int y = 0; y < IMG_SIZE; y++){
		for(int x = 0; x < IMG_SIZE; x++){
			m_std[y][x] = sqrt(m_std[y][x]/((float)m_trainDB.size()-1));
		}
	}
}

void dataReader::apply_nor_into_trainDB(){
	printf("[Normalization] Train DB\n");
	assert(m_trainDB.size() > 0);

	for(auto iter = m_trainDB.begin(); iter != m_trainDB.end(); iter++){
		for(int y = 0; y < IMG_SIZE; y++){
			for(int x = 0; x < IMG_SIZE; x++){
				if(m_std[y][x] > 1e-10){
					(*iter)->nor_data.twoD[y][x] = (float)((*iter)->data[y][x] - m_mean[y][x])/m_std[y][x];
				} else{
					(*iter)->nor_data.twoD[y][x] = 0.0f;
				}
			}
		}
	}
}

void dataReader::apply_nor_into_testDB(){
	printf("[Normalization] Test DB\n");
	assert(m_testDB.size() > 0);

	for(auto iter = m_testDB.begin(); iter != m_testDB.end(); iter++){
		for(int y = 0; y < IMG_SIZE; y++){
			for(int x = 0; x < IMG_SIZE; x++){
				if(m_std[y][x] > 1e-10){
					(*iter)->nor_data.twoD[y][x] = (float)((*iter)->data[y][x] - m_mean[y][x])/m_std[y][x];
				} else{
					(*iter)->nor_data.twoD[y][x] = 0.0f;
				}
			}
		}
	}
}

m_data* dataReader::get_mnist_db(unsigned idx, bool trainDB){
	std::vector<m_data *>::iterator iter;
	if(trainDB){
		assert(idx < m_trainDB.size());
		iter = m_trainDB.begin() + idx;
	}else{
		assert(idx < m_testDB.size());
		iter = m_testDB.begin() + idx;
	}
	return (*iter);
}

unsigned dataReader::get_mnist_db_size(bool trainDB){
	if(trainDB){
		return m_num_trainDB;
	}
	return m_num_testDB;
}
