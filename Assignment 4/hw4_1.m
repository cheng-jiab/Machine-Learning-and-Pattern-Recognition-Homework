clear all; close all;

N_train = 1000;
N_valid = 10000;

[x_train,y_train] = exam4q1_generateData(N_train);
[x_valid,y_valid] = exam4q1_generateData(N_valid);






function [x,y] = exam4q1_generateData(N)
close all,
x = gamrnd(3,2,1,N);
z = exp((x.^2).*exp(-x/2));
v = lognrnd(0,0.1,1,N);
y = v.*z;
figure(1), plot(x,y,'.'),
xlabel('x'); ylabel('y');
end