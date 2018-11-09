function neural_network = initial_NN()
global input_num hidden_num output_num 

W_inhidden = 2 * rand(input_num, hidden_num) - 1;    %Ȩֵ[-1,1]
W_hiddenout = 2 * rand(hidden_num, output_num) - 1;
trans1 = W_inhidden(1,:);
for i=2:input_num
    trans1 = [trans1 W_inhidden(i,:)];
end
trans2 = W_hiddenout(1,:);
for i=2:hidden_num
    trans2 = [trans2 W_hiddenout(i,:)];
end
neural_network = [trans1 trans2];
end
