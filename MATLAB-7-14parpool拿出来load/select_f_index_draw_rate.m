function  [select_index_x,select_index_f] = select_f_index_draw_rate(temp,f,index_x,d,index_f,W_inhidden,W_hiddenout)%temp初始，f是初始位置的适应值，index_x是排序之后的位置序号index_f是适应值排序的序号
global section 
[row,column] = size(index_x);     %行,列
if row<30
    randsel = randperm(row,1);
    boundary(1) = temp(index_x(1),d);
    for i=1:section-1               %中间临界平均分
        boundary(i+1) = (temp(index_x(i),d) + temp(index_x(i+1,1),d))/2;
    end
    boundary(section+1) =  temp(index_x(end),d);
    
    select_index_x = index_x(randsel);
elseif mod(row,section) == 0
    
    a_index_section = reshape(index_x,floor(row/section),section); %位置序号分成10个区间，1个区间有100个/10个/1个;100*10
    
    for j=1:row
        temp_f(j) = f(index_f(j));
    end
    
    a_section_fit = reshape(temp_f,floor(row/section),section);  %适应值f所对应的分隔区间 100*10,
    a_section_fit_index = reshape(index_f,floor(row/section),section);%把排好序的fitness索引分区间 100*10
    
    for i=1:section                             %每组取fitness的最大值、最小值和中值
        [v1(i),index1(i)] = max(a_section_fit(:,i)); %在此区间里f最大的，所对应的下标
        [v2(i),index2(i)] = min(a_section_fit(:,i)); %最小的
        [v3 index3(i,:)] = sort(a_section_fit(:,i));
        mid(i) = floor(median(index3(i,:)));                %最中的
        
        max_index(i) = a_section_fit_index(index1(i),i); %最值所对应的f的排序序号
        min_index(i) = a_section_fit_index(index2(i),i);
        mid_index(i) = a_section_fit_index(mid(i),i);
        
        section_fit_index_sort(i,:) = [max_index(i)/row min_index(i)/row mid_index(i)/row]; %1个区间的3个序号
    end
    %记录边界的值(位置)  中间边界取平均
    boundary(1) = temp(a_index_section(1),d);
    for i=1:section-1
        boundary(i+1) = (temp(a_index_section(end,i),d) + temp(a_index_section(1,i+1),d))/2;
    end
    boundary(section+1) =  temp(a_index_section(end,end),d);
    
    every_rate = nn_compute(section_fit_index_sort,W_inhidden,W_hiddenout);
    
    %轮盘赌选择落到哪一个区间上
    sum_rate = cumsum(every_rate);  %积累概率
    [next_boundary,a_section_index] = select_action(sum_rate,boundary);  %2个边界值,2个对应的下标(1-11)
    %     draw1_boundary = next_boundary;   %第一次分的边界
    
    select_index_x = a_index_section(:,a_section_index(1));   %选出新的位置在哪一个区间里的序号，是100/10*1的
    select_index_f = a_section_fit_index(:,a_section_index(1));
else
    select_index_x = index_x(randperm(row,1));
    select_index_f = index_f(randperm(row,1));
end
