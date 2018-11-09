
global initial_flag MAX_FES
global INIT_MIN INIT_MAX section positionDim best_value
global input_num hidden_num output_num FUNNUM fitcount benchmark_number nn_num

initial_flag = 0;

section = 10;                        %ÿ�η�10��
benchmark_number = 10;               %10�����Ժ���
nn_num = 10;                         %10��������
MAX_GENERATION = 1000;              %GA10��

best_value = 0;
positionDim = 2;                      %�����ά��
fitcount = 0;
% PLOT_INTERVAL = POP_SIZE*2;
MAX_FES = 2000000;%200��FE
% MAX_GENERATION = floor(MAX_FES/POP_SIZE)+1;              %�ܽ�������

num_recurrent = 100;

pc=0.7;                        %�������
pm=0.3;                        %�������

input_num = 30;
hidden_num = 40;
output_num = 10;

[neural_network,record_nn_fitness_sort_index] = initial_NN();

for g=1:MAX_GENERATION
    selnn = selection_tournment(neural_network,record_nn_fitness_sort_index(:,benchmark_number+1));
    cronn= crossover(selnn,pc);                      %����
    newnn = mutation(cronn,pm);                      %����
    neural_network = newnn;
    
    for nn=1:nn_num
        re_convert1 = neural_network(nn,1:input_num*hidden_num);
        re_convert2 = neural_network(nn,(input_num*hidden_num)+1:end);
        W_inhidden = reshape(re_convert1,40,30)';
        W_hiddenout = reshape(re_convert2,10,40)';
        FUNNUM = 1;                          %���Եĺ������
        for fun_num=1:benchmark_number
            nn_fitness(nn,fun_num) = dif_nn(W_inhidden,W_hiddenout);  %1�������������к������в�ͬ��fitness������λ��x������ģ�
%             nn_fitness(nn,fun_num) = dif_nn(W_inhidden,W_hiddenout);  %1�������������к������в�ͬ��fitness������λ��x������ģ�
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
    for fun_num=1:benchmark_number         %10�����Ժ���ʵ��
        [sort_fit,record_sort_index(:,fun_num)]= sort(nn_fitness(:,fun_num));%��20��������һ�������µ�fitness��������
        for nn=1:nn_num
            pre_th = record_sort_index(nn,fun_num);  %������1-20�����������ԭ�±�
            record_nn_fitness_sort_index(pre_th,fun_num) = nn;  %ԭ�������ڵ�fun_num�������µ�����
        end
    end
    for i=1:nn_num
        record_nn_fitness_sort_index(i,benchmark_number+1) = mean(record_nn_fitness_sort_index(i,1:benchmark_number)); %��benchmark_number+1����ƽ������ֵ~nn��Ӧֵ
        fprintf('��%d��GA����%d��������, ��%d��������ƽ������Ϊ�� %e \n',g,i,benchmark_number,record_nn_fitness_sort_index(i,benchmark_number+1));
    end
    min_fit(g)=min(record_nn_fitness_sort_index(:,benchmark_number+1));
    fprintf('��%d��GA����%d����������С����Ϊ�� %e \n',g,benchmark_number,record_nn_fitness_sort_index(i,benchmark_number+1));

end