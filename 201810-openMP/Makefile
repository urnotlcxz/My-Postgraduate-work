cc=gcc
target=main
obj=main.o fitnessf.o cec17_test_func.o
$(target):$(obj)
	$(cc) $(obj) -Wall -o $(target) -lm -fopenmp
main.o:main.c
	$(cc) -c -g main.c -lm -fopenmp
fitnessf.o:fitnessf.c
	$(cc) -c -g fitnessf.c -lm -fopenmp
cec17_test_func.o:cec17_test_func.c
	$(cc) -c -g cec17_test_func.c -lm -fopenmp
.PHONY:clean
	$(RM) *.o $(target)

