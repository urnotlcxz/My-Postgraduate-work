%���̶�ѡ������
function [next_boundary,a_section_index]= select_action(sum_rate,boundary)
global section
r = rand();
if r < sum_rate(1)
    next_boundary = [boundary(1) boundary(2)];
    a_section_index = [1 2];
else
    for j=1:section-1
        if(r>=sum_rate(j) && r<sum_rate(j+1))
            next_boundary = [boundary(j+1) boundary(j+2)]; %ȡ������߽緶Χ
            a_section_index = [j+1 j+2];    %�߽�����1-11֮��
        end
    end
end
end

%         C���԰汾
%         int RWS() {
%             m = 0;
%             r = Random(0, 1); //rΪ0��1�������
%             for (i = 1; i <= N; i++) {
%             /**
%              * �������������m~m+P[i]������Ϊѡ����i�����i��ѡ�еĸ�����P[i]
%              */
%                 m = m + P[i];
%                 if (r <= m) return i;
%             }
%         }
