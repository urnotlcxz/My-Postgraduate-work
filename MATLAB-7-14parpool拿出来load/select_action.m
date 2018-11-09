%轮盘赌选择区间
function [next_boundary,a_section_index]= select_action(sum_rate,boundary)
global section
r = rand();
if r < sum_rate(1)
    next_boundary = [boundary(1) boundary(2)];
    a_section_index = [1 2];
else
    for j=1:section-1
        if(r>=sum_rate(j) && r<sum_rate(j+1))
            next_boundary = [boundary(j+1) boundary(j+2)]; %取出区间边界范围
            a_section_index = [j+1 j+2];    %边界的序号1-11之间
        end
    end
end
end

%         C语言版本
%         int RWS() {
%             m = 0;
%             r = Random(0, 1); //r为0至1的随机数
%             for (i = 1; i <= N; i++) {
%             /**
%              * 产生的随机数在m~m+P[i]间则认为选中了i，因此i被选中的概率是P[i]
%              */
%                 m = m + P[i];
%                 if (r <= m) return i;
%             }
%         }
