cpp_objects = main.o filterbank.o
cuda_objects = fb_multi_channel_Impl.o 


fb: main.o filterbank.o fb_multi_channel_Impl.o 
	g++ -dlink main.o filterbank.o fb_multi_channel_Impl.o -lcufft -lcudart -o fb

main.o : ./source/main.cpp ./include/main.hpp
	g++ -c ./source/main.cpp

filterbank.o : ./source/filterbank.cpp ./include/filterbank.hpp
	g++ -c ./source/filterbank.cpp

fb_multi_channel_Impl.o : ./source/fb_multi_channel_Impl.cu ./include/fb_multi_channel_Impl.cuh
	nvcc -arch=sm_60 -c ./source/fb_multi_channel_Impl.cu -lcufft -lcudart 

clean:
	rm -rf *.o fb
