function  [select_index_x,select_index_f] = select_f_index_draw_rate(temp,f,index_x,d,index_f,W_inhidden,W_hiddenout)%temp��ʼ��f�ǳ�ʼλ�õ���Ӧֵ��index_x������֮���λ�����index_f����Ӧֵ��������
global section 
[row,column] = size(index_x);     %��,��
if row<30
    randsel = randperm(row,1);
    boundary(1) = temp(index_x(1),d);
    for i=1:section-1               %�м��ٽ�ƽ����
        boundary(i+1) = (temp(index_x(i),d) + temp(index_x(i+1,1),d))/2;
    end
    boundary(section+1) =  temp(index_x(end),d);
    
    select_index_x = index_x(randsel);
elseif mod(row,section) == 0
    
    a_index_section = reshape(index_x,floor(row/section),section); %λ����ŷֳ�10�����䣬1��������100��/10��/1��;100*10
    
    for j=1:row
        temp_f(j) = f(index_f(j));
    end
    
    a_section_fit = reshape(temp_f,floor(row/section),section);  %��Ӧֵf����Ӧ�ķָ����� 100*10,
    a_section_fit_index = reshape(index_f,floor(row/section),section);%���ź����fitness���������� 100*10
    
    for i=1:section                             %ÿ��ȡfitness�����ֵ����Сֵ����ֵ
        [v1(i),index1(i)] = max(a_section_fit(:,i)); %�ڴ�������f���ģ�����Ӧ���±�
        [v2(i),index2(i)] = min(a_section_fit(:,i)); %��С��
        [v3 index3(i,:)] = sort(a_section_fit(:,i));
        mid(i) = floor(median(index3(i,:)));                %���е�
        
        max_index(i) = a_section_fit_index(index1(i),i); %��ֵ����Ӧ��f���������
        min_index(i) = a_section_fit_index(index2(i),i);
        mid_index(i) = a_section_fit_index(mid(i),i);
        
        section_fit_index_sort(i,:) = [max_index(i)/row min_index(i)/row mid_index(i)/row]; %1�������3�����
    end
    %��¼�߽��ֵ(λ��)  �м�߽�ȡƽ��
    boundary(1) = temp(a_index_section(1),d);
    for i=1:section-1
        boundary(i+1) = (temp(a_index_section(end,i),d) + temp(a_index_section(1,i+1),d))/2;
    end
    boundary(section+1) =  temp(a_index_section(end,end),d);
    
    every_rate = nn_compute(section_fit_index_sort,W_inhidden,W_hiddenout);
    
    %���̶�ѡ���䵽��һ��������
    sum_rate = cumsum(every_rate);  %���۸���
    [next_boundary,a_section_index] = select_action(sum_rate,boundary);  %2���߽�ֵ,2����Ӧ���±�(1-11)
    %     draw1_boundary = next_boundary;   %��һ�ηֵı߽�
    
    select_index_x = a_index_section(:,a_section_index(1));   %ѡ���µ�λ������һ�����������ţ���100/10*1��
    select_index_f = a_section_fit_index(:,a_section_index(1));
else
    select_index_x = index_x(randperm(row,1));
    select_index_f = index_f(randperm(row,1));
end
