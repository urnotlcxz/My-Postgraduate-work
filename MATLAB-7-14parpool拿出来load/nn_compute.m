function output = nn_compute(section_fitness_sort,W_inhidden,W_hiddenout)   %通过普通的前馈神经网络
global input_num hidden_num output_num

x=section_fitness_sort; %10*3
[d,M,D] = size(x);

for i = 1:hidden_num
    sumv = 0;
    for j = 1:input_num
        sumv = sumv + W_inhidden(input_num*(i-1)+j) * x(j);
    end
    wh_bias(i) = 2*rand-1;
    hiddenoutputs(i) = sumv+wh_bias(i);
end
for i = 1:output_num
    sumv = 0;
    for j = 1:hidden_num
        sumv = sumv + W_hiddenout(hidden_num*(i-1)+j)*hiddenoutputs(j);
    end
    wo_bias(i) = 2*rand-1;
    sumv = sumv+wo_bias(i);
    newx(i) = exp(sumv);
end
%clear hiddenoutputs;
for i = 1:output_num
    output(i) = newx(i)/sum(newx);
end
end

