clear all;
close all;
warning('off')
parfevalOnAll(gcp(), @warning, 0, 'off', 'MATLAB:singularMatrix');
n=2;
C=7;
N_o=10^5;
[N_train] = [10^2, 10^3, 10^4, 10^5];
alpha = (ones(C,1)/C)';
    BIC_100 = [];
    BIC_1k = [];
    BIC_10k = [];
    BIC_100k = [];
    KF_100 = [];
    KF_1k = [];
    KF_10k = [];
    KF_100k = [];
miu = cumsum(ones(1,C)/C);
for i=1:C
    mu(:,i)= [20*miu(i) 0]; %adjustable
    sigma(:,:,i) = 1*rand(1)*eye(n);
end

X_o = gmm_gen(n,N_o, alpha, mu,sigma);
figure(1)
plot(X_o(1,:),X_o(2,:),'o',mu(1,:),mu(2,:),'+')

E=100;
hist_BIC = zeros(E,length(N_train));
hist_KF = zeros(E,length(N_train));

for e =1:E
    e
    clearvars sigma_e;
    clearvars mu_e;
    clearvars alpha_e;
    level = ceil(4*rand(1));
    %level=1
    N = N_train(level)
    X = X_o(:,ceil(N*rand(1,N)));
    
    %-------------BIC
    %kfold
    npercept = 5;
    K=10; %---------
    
    maxM=14; %----------
    tic
    for M = 1:maxM
        
        nParams(1,M) = (M-1)+ n*M + M*(n+nchoosek(2,2));
       [alpha_e,mu_e,sigma_e] = EMforGmm(M,N,X);
        neg2loglike(1,M) = -2*sum(log(eval_gmm(X,alpha_e,mu_e,sigma_e)));
        BIC(1,M) = neg2loglike(1,M) + nParams(1,M)*log(N);


        %kfold
        clearvars dummy;
        dummy = ceil(linspace(0,N,K+1));
        
        parfor k = 1:K %enable parrallel
           
           %part_ind(k,:)=[dummy(k)+1, dummy(k+1)];
           K_valid = X(:,[dummy(k)+1:dummy(k+1)]);
           train_ind = setdiff([1:N],[dummy(k)+1:dummy(k+1)]);
           K_train = X(:,train_ind);
    %estimate parameter
            
           [alpha_e,mu_e,sigma_e] = EMforGmm(M,length(K_train),K_train);
           loglike(k,:) = sum(log(eval_gmm(K_valid,alpha_e,mu_e,sigma_e)));
        end
        
        aveloglike(1,M) = mean(loglike);
    end
    toc
    
    [~, BIC_M] = min(BIC);
    [~, KF_M] = max(aveloglike);

    

    if level==1
        BIC_100 = [BIC_100, BIC_M]
        KF_100 = [KF_100, KF_M]
    elseif  level==2
        BIC_1k = [BIC_1k, BIC_M]
        KF_1k = [KF_1k, KF_M]
    elseif level==3
        BIC_10k = [BIC_10k, BIC_M]
        KF_10k = [KF_10k, KF_M]
    elseif level==4
        BIC_100k = [BIC_100k, BIC_M]
        KF_100k = [KF_100k, KF_M]
    end
    
    %hist_BIC(e,level)= BIC_M
    %hist_KF(e,level) = KF_M
    


end

max1 = [max(BIC_100) max(BIC_1k) max(BIC_10k) max(BIC_100k)];
avg1 = [median(BIC_100) median(BIC_1k) median(BIC_10k) median(BIC_100k)];
low1 = [min(BIC_100) min(BIC_1k) min(BIC_10k) min(BIC_100k)];
figure(2)
bar([1:4],[max1; avg1;low1]);
legend('max','median','min');

max2 = [max(KF_100) max(KF_1k) max(KF_10k) max(KF_100k)];
avg2 = [median(KF_100) median(KF_1k) median(KF_10k) median(KF_100k)];
low2 = [min(KF_100) min(KF_1k) min(KF_10k) min(KF_100k)];
figure(3)
bar([1:4],[max2; avg2;low2]);
legend('max','median','min');

function [alpha_e,mu,Sigma] = EMforGmm(M,N,x)

regWeight = 1e-10;
delta = 1e-2;
alpha_e = ones(1,M)/M;
shuffle = randperm(N);
mu= x(:,shuffle(1:M));
[~, centroidlabel] = min(pdist2(mu',x'),[],1);
for m= 1:M
    Sigma(:,:,m) = cov(x(:,find(centroidlabel==m))')+regWeight*eye(2,2);
    
end

%maximum
t=0;
converged = 0;
while ~converged
    
   
   for l = 1:M
       temp(l,:) = repmat(alpha_e(l),1,N).*eval_g(x,mu(:,l),Sigma(:,:,l));
   end
   
   
   plgivenx = temp./sum(temp,1);
   alphaNew = mean(plgivenx,2);
   w = plgivenx ./repmat(sum(plgivenx,2),1,N);
   muNew = x*w';
   
   for l=1:M
      v= x-repmat(muNew(:,l),1,N);
      u = repmat(w(l,:),2,1).*v;
      SigmaNew(:,:,l) = u*v' +regWeight*eye(2,2);  
   end
   
   Dalpha = sum(abs(alphaNew-alpha_e));
   Dmu = sum(sum(abs(muNew-mu)));
   DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
   converged = ((Dalpha+Dmu+DSigma)<delta);
   alpha_e = alphaNew;
   mu = muNew;
   Sigma = SigmaNew;
   t=t+1;
   if t == 150
       converged = 1;
   end
end

end

function s=eval_gmm(x,alpha,mu0,Sigma0)
[n,N] = size(x);
g = zeros(length(alpha),N);

for i= 1:length(alpha)   
    C = ((2*pi)^n * det(Sigma0(:,:,i)))^(-1/2);
    E = -0.5*sum((x-repmat(mu0(:,i),1,N)).*(inv(Sigma0(:,:,i))*(x-repmat(mu0(:,i),1,N))),1);
    g(i,:)= C*exp(E);
end

g;
s = alpha'* g;

end


function xgmm = gmm_gen(n,N,alpha, mu0, sigma0)
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N);
xgmm = zeros(n,N);
for m = 1:length(alpha)
    ind = find((cum_alpha(m)<u)&(u <=cum_alpha(m+1)));
    xgmm(:,ind)=mvnrnd(mu0(:,m),sigma0(:,:,m),length(ind))';
    
end
end


function g = eval_g(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
%invSigma = Sigma\eye(size(Sigma)) ;
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end