
global initial_flag MAX_FES
global INIT_MIN INIT_MAX section positionDim best_value
global input_num hidden_num output_num FUNNUM fitcount benchmark_number nn_num

initial_flag = 0;

section = 10;                        %每次分10份
benchmark_number = 10;               %10个测试函数
nn_num = 10;                         %10个神经网络
MAX_GENERATION = 1000;              %GA10代

best_value = 0;
positionDim = 2;                      %问题的维数
fitcount = 0;
% PLOT_INTERVAL = POP_SIZE*2;
MAX_FES = 2000000;%200万FE
% MAX_GENERATION = floor(MAX_FES/POP_SIZE)+1;              %总进化次数

num_recurrent = 100;

pc=0.7;                        %交叉概率
pm=0.3;                        %变异概率

input_num = 30;
hidden_num = 40;
output_num = 10;

[neural_network,record_nn_fitness_sort_index] = initial_NN();

for g=1:MAX_GENERATION
    selnn = selection_tournment(neural_network,record_nn_fitness_sort_index(:,benchmark_number+1));
    cronn= crossover(selnn,pc);                      %交叉
    newnn = mutation(cronn,pm);                      %变异
    neural_network = newnn;
    
    for nn=1:nn_num
        re_convert1 = neural_network(nn,1:input_num*hidden_num);
        re_convert2 = neural_network(nn,(input_num*hidden_num)+1:end);
        W_inhidden = reshape(re_convert1,40,30)';
        W_hiddenout = reshape(re_convert2,10,40)';
        FUNNUM = 1;                          %测试的函数标号
        for fun_num=1:benchmark_number
            nn_fitness(nn,fun_num) = dif_nn(W_inhidden,W_hiddenout);  %1个神经网络在所有函数下有不同的fitness（根据位置x来计算的）
%             nn_fitness(nn,fun_num) = dif_nn(W_inhidden,W_hiddenout);  %1个神经网络在所有函数下有不同的fitness（根据位置x来计算的）
          FUNNUM = FUNNUM+1;
        end
        trans1 = W_inhidden(1,:);
        for i=2:input_num
            trans1 = [trans1 W_inhidden(i,:)];
        end
        trans2 = W_hiddenout(1,:);
        for i=2:hidden_num
            trans2 = [trans2 W_hiddenout(i,:)];
        end
        trans = [trans1 trans2];
        neural_network(nn,:) = trans;
        clear trans trans1 trans2
    end
    for fun_num=1:benchmark_number         %10个测试函数实验
        [sort_fit,record_sort_index(:,fun_num)]= sort(nn_fitness(:,fun_num));%对20个网络在一个函数下的fitness进行排序
        for nn=1:nn_num
            pre_th = record_sort_index(nn,fun_num);  %排名第1-20个的神经网络的原下标
            record_nn_fitness_sort_index(pre_th,fun_num) = nn;  %原神经网络在第fun_num个函数下的排名
        end
    end
    for i=1:nn_num
        record_nn_fitness_sort_index(i,benchmark_number+1) = mean(record_nn_fitness_sort_index(i,1:benchmark_number)); %第benchmark_number+1列是平均排名值~nn适应值
        fprintf('第%d代GA，第%d个神经网络, 在%d个函数的平均排名为： %e \n',g,i,benchmark_number,record_nn_fitness_sort_index(i,benchmark_number+1));
    end
    min_fit(g)=min(record_nn_fitness_sort_index(:,benchmark_number+1));
    fprintf('第%d代GA，在%d个函数的最小排名为： %e \n',g,benchmark_number,record_nn_fitness_sort_index(i,benchmark_number+1));

end