CC=nvcc 

blur.out:  clockMeasure.h ppm.h main.cu
	$(CC) -o blur.out main.cu

clean:
	rm -rf blur.out *.ppm
