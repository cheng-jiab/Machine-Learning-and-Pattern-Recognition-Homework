close all;
clear all;

n = 7;
N_train = 100;
N_valid = 10000;

a = ceil(n*rand(n,1));
Mu = 3*rand(n,1); %adjust
Sigma = eye(n)+ 0.2*rand(n,n); %adujust
Sigma = Sigma'*Sigma;

%alpha = 1;
alpha_loop = 10.^(-3:0.05:3)*trace(Sigma);
%alpha_plot = round(1:length(alpha_loop)/7:length(alpha_loop))

Mu_z = zeros(n,1);
%Sigma_z = alpha * eye(n);

Mu_v = 0;
Sigma_v = 1;

%beta = 1;
beta_loop =logspace(-4,3);
    
Mu_w = 0;
%Sigma_w = beta*eye(n);

%loop

for i = 1:length(alpha_loop)
    Sigma_z = alpha_loop(i) * eye(n);
    x_train = mvnrnd(Mu,Sigma, N_train)';
    z_train = mvnrnd(Mu_z,Sigma_z, N_train)';
    v_train = Mu_v + Sigma_v*randn(N_train,1);
    
    y_train = (a'*(x_train+z_train))'+v_train;
    
    x_valid = mvnrnd(Mu,Sigma, N_valid)';
    z_valid = mvnrnd(Mu_z,Sigma_z, N_valid)';
    v_valid = Mu_v + Sigma_v*randn(N_valid,1);
    
    y_valid = (a'*(x_valid+z_valid))'+v_valid;
    
    for j =1:length(beta_loop)
           
        %start k fold
        K = 10;
        dummy = ceil(linspace(0,N_train,K+1));
        for k = 1:K
            K_x_valid = x_train(:,[dummy(k)+1:dummy(k+1)]);
            K_y_valid = y_train([dummy(k)+1:dummy(k+1)]);
            train_ind = setdiff([1:N_train],[dummy(k)+1:dummy(k+1)]);
            K_x_train = x_train(:,train_ind);
            K_y_train = y_train(train_ind);
            
            A = [ones(1,length(K_x_train)); K_x_train];
            
            w_est = inv((A*A')+ Sigma_v^2/beta_loop(j)^2*eye(size(A,1)))*(K_y_train'*A')';
            
            valid_A = [ones(1,length(K_x_valid)); K_x_valid];
            
            MSE_K(k) = mean((K_y_valid'-w_est'*valid_A).^2);
                  
        end
        avgMSE(j) = mean(MSE_K);
                
    end
    [minMSE,min_ind] = min(avgMSE)
    
    A=[ones(1,N_train); x_train];
    w_est = inv((A*A')+ Sigma_v^2/beta_loop(min_ind)^2*eye(size(A,1)))*(y_train'*A')';
    log_MSE_train = (mean((y_train'-w_est'*A).^2));
    
    A_valid = [ones(1,N_valid); x_valid]
    MSE_valid = mean((y_valid'-w_est'*A_valid).^2);
    
    pxgivenwx = -2*log(eval_g(y_valid',(w_est'*A_valid),Sigma_v));
    
    %result
    MSE(i) = MSE_valid
    log_MSE_train(i) = log_MSE_train
    beta_s(i) = beta_loop(min_ind)
    alpha_a(i) = alpha_loop(i)
    
    
end

figure(1)
loglog(alpha_a,beta_s,'o')
xlabel('Alpha');
ylabel('Beta');
title('Alpha vs. Selected Beta')

figure(2)
loglog(alpha_a,MSE,'o')
xlabel('Alpha');
ylabel('MSE');
title('Alpha vs. Minimum Square Error')

figure(3)
loglog(1:length(beta_loop),avgMSE,'o')
function g = eval_g(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-mu).*(invSigma*(x-mu)),1);
g = C*exp(E);
end



