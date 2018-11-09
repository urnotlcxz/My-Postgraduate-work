function val=fitnessf(x,funnum,o,M,bias)
% global fitcount
% fitcount=fitcount+1;

[m, D]=size(x);
switch funnum
    case 1 % Shifted Sphere Function 
         val=benchmark_func(x,1,o,M,bias);
%          val=0;
%         for i=1:D
%             val=val+(floor(x(:,i)+0.5).*floor(x(:,i)+0.5));
%         end
    case 2 % Shifted Schwefel's Problem 1.2	
         val=benchmark_func(x,2,o,M,bias);
    case 3 % Shifted Rotated High Conditioned Elliptic Function
        val=benchmark_func(x,3,o,M,bias);
    case 4 % %Step  F4
        val=0;
        %          for i=1:D
        %              val=val+(floor(x(i)+0.5)*floor(x(i)+0.5));
        %          end
        for i=1:D
            val=val+(floor(x(:,i)+0.5).*floor(x(:,i)+0.5));
        end
    case 5 %Shifted Rotated Weierstrass Function
        val=benchmark_func(x,11,o,M,bias);
    case 6 %Shifted Rosenbrock's Function
        val=benchmark_func(x,6,o,M,bias);
    case 7 %Shifted Rotated Griewank's Function
        val=benchmark_func(x,7,o,M,bias);
    case 8 %Shifted Rotated Ackley's Function with Global Optimum on Bounds
        val=benchmark_func(x,8,o,M,bias);
    case 9 %Shifted Rastrign's Function
         val=benchmark_func(x,9,o,M,bias);
    case 10 %Expanded Rotated Extended Scaffer's F6 	
        val=benchmark_func(x,14,o,M,bias);%
end
end