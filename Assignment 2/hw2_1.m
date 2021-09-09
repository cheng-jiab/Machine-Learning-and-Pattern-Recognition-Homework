% Question 1
% Init
clear all;
close all;
n=2;
N_train = [100,1000,10000];
N_valid = 20000;
P = [0.6,0.4];
mu1 = [3;2];
C1 = [2 0; 0 2];

mu0 = [5 0; 0 4];
C0(:,:,1) = [4 0; 0 2]; 
C0(:,:,2) = [1 0; 0 3];

alpha = [0.5,0.5];
figure(1)
[X_train_100, label100] = generate_data(n, N_train(1),P,mu1,C1,alpha,mu0,C0,1);
[X_train_1000, label1000] = generate_data(n, N_train(2),P,mu1,C1,alpha,mu0,C0,2);
[X_train_10000, label10000] = generate_data(n, N_train(3),P,mu1,C1,alpha,mu0,C0,3);

[X_valid, vlabel] = generate_data(n,N_valid,P,mu1,C1,alpha,mu0,C0,4);

%plot(X_train_100(1,:),X_train_100(2,:),'o')

% Part 1
%score = eval_g(X_train_100,mu1,C1)
figure_n = 1;
figure(2)
min_error = PClassifier(X_valid,vlabel,P,mu1,C1,alpha,mu0,C0,figure_n)

%-------------------_????_------------------
%create grid?
%h_grid = linspace(floor(min(X_valid(1,:)))-2,ceil(max(X_valid(1,:)))+2);
%v_grid = linspace(floor(min(X_valid(2,:)))-2,ceil(max(X_valid(2,:)))+2);
%[h,v]= meshgrid(h_grid,v_grid);

%calculate score for gird point
%Score_grid = log(eval_g([h(:)';v(:)'],mu1,C1))-log(eval_gmm([h(:)';v(:)'],alpha,mu0,C0));
%S_grid = reshape(Score_grid,length(h_grid),length(v_grid));

%figure(2)
%plot_classified_data



% Part2
%Train10000
[P_est,mu1_est,C1_est,alpha_e,mu0_est,C0_est]= Param_est(X_train_10000,label10000);
figure(3)
figure_n = 1
alphaT = alpha_e';
min_error = PClassifier(X_valid,vlabel,P_est,mu1_est,C1_est,alphaT,mu0_est,C0_est,figure_n)
%train1000
[P_est,mu1_est,C1_est,alpha_e,mu0_est,C0_est]= Param_est(X_train_1000,label1000)

figure_n = 2
alphaT = alpha_e';
min_error = PClassifier(X_valid,vlabel,P_est,mu1_est,C1_est,alphaT,mu0_est,C0_est,figure_n)
%train100
[P_est,mu1_est,C1_est,alpha_e,mu0_est,C0_est]= Param_est(X_train_100,label100)

figure_n = 3
alphaT = alpha_e';
min_error = PClassifier(X_valid,vlabel,P_est,mu1_est,C1_est,alphaT,mu0_est,C0_est,figure_n)

%-------------
%part 3
%Linear
%figure(4)
[error_L10k,error_Q10k] = logistic_classifier(X_train_10000,label10000,N_valid, X_valid, vlabel,4,P)

[error_L1k,error_Q1k] = logistic_classifier(X_train_1000,label1000,N_valid, X_valid, vlabel,5,P)

[error_L,error_Q] = logistic_classifier(X_train_100,label100,N_valid, X_valid, vlabel,6,P)


function cost = cost_func(theta, x, label, N)
h=1 ./ (1+exp(-x*theta));
cost = (-1/N)*((sum(label'*log(h)))+(sum((1-label)'*log(1-h))));

end

%---------------------------
function [P_est,mu1_est,C1_est,alpha_e,mu0_est,C0_est] = Param_est(X,label)
P_est = [1-(sum(label)/length(label)), sum(label)/length(label)];
M = 1; 

x= X(:,label==1);
N= length(x);
[aa,mu1_est,C1_est] = EMforGmm(M,N,x)

M = 2; 

x= X(:,label==0);
N= length(x);
[alpha_e,mu0_est,C0_est]=EMforGmm(M,N,x)

end


%---------------------------------
% Generate true labels
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
   t=t+1    
end

end

%draw plot
function min_error = PClassifier(X,label,P,mu1,C1,alpha,mu0,C0,figure_n)
Score = log(eval_g(X,mu1,C1))-log(eval_gmm(X,alpha,mu0,C0));
tau = log(sort(Score(Score >=0)));
mid_tau =[tau(1)-1, tau(1:end-1)+diff(tau)./2 tau(end)+1];
for i = 1:length(mid_tau)
    decision = (Score>=mid_tau(i));
    pFA(i) = sum(decision==1 & label== 0)/length(label(label==0));
    pCD(i) = sum(decision==1 & label== 1)/length(label(label==1));
    pE(i) = pFA(i)*P(1)+(1-pCD(i))*P(2);
    
end
%find the minimal pe
[min_error,min_index] = min(pE);
min_decision = (Score >=mid_tau(min_index));
min_FA = pFA(min_index);
min_CD = pCD(min_index);

subplot(1,3,figure_n)
plot(pFA,pCD,'-',min_FA,min_CD,'o')
end

%part3
function [error_L,error_Q] = logistic_classifier(X_train,label_t,N_valid, X_valid, vlabel,figure_n,P)
n=2;

x_L = [ones(length(X_train),1) X_train'];
initial_theta_L = zeros(n+1,1);
label = double(label_t)';
[theta_L, cost_L] = fminsearch(@(t)(cost_func(t,x_L,label,length(X_train))),initial_theta_L);
%validate
valid_L = [ones(N_valid,1) X_valid'];
decision_L = valid_L*theta_L>=0;
pFA=length(find(decision_L'==1 & vlabel==0))/length(find(vlabel ==0));
pMD=length(find(decision_L'==0 & vlabel==1))/length(find(vlabel ==1));
error_L = pFA*P(1)+pMD*P(2);
%plot
plot_x1 = [min(x_L(:,2))-2, max(x_L(:,2))+2];
plot_x2 = (-1./theta_L(3).* theta_L(2).*plot_x1+theta_L(1));
bound = [plot_x1;plot_x2];

figure(figure_n);
subplot(2,1,1)
plot(x_L(label==0,2),x_L(label==0,3),'o',x_L(label==1,2),x_L(label==1,3),'+');hold on;
plot(bound(1,:),bound(2,:));

%Q
x_Q = [ones(length(X_train),1) X_train',(X_train(1,:).*X_train(2,:))', (X_train.^2)'];
initial_theta_Q = zeros(6,1);

[theta_Q, cost_Q] = fminsearch(@(t)(cost_func(t,x_Q,label,length(X_train))),initial_theta_Q);
%validate
valid_Q =[ones(length(X_valid),1) X_valid',(X_valid(1,:).*X_valid(2,:))', (X_valid.^2)'];
decision_Q = valid_Q *theta_Q >=0;
pFA=length(find(decision_Q'==1 & vlabel==0))/length(find(vlabel ==0));
pMD=length(find(decision_Q'==0 & vlabel==1))/length(find(vlabel ==1));
error_Q = pFA*P(1)+pMD*P(2);

hgrid = linspace(min(x_Q(:,2))-6, max(x_Q(:,2))+6,20);
vgrid = linspace(min(x_Q(:,3))-6, max(x_Q(:,3))+6,20);
z= zeros(length(hgrid),length(vgrid));
for i=1:length(hgrid)
   for j=1:length(vgrid)
       xbound=[1, hgrid(i) vgrid(j) hgrid(i)^2 hgrid(i)*vgrid(j) vgrid(j)^2];
       z(i,j) = xbound* theta_Q;
   end
end
gridscore = z';

bound = [hgrid;vgrid;gridscore];
subplot(2,1,2)
plot(x_Q(label==0,2),x_Q(label==0,3),'o',x_Q(label==1,2),x_Q(label==1,3),'+');hold on;
contour(bound(1,:),bound(2,:),bound(3:end,:),[0,0]);
end




function [x, label] = generate_data(n, N, P, mu1, sigma1,alpha,mu0,sigma0,i)
x = zeros(n,N);
label = (rand(1,N)>=P(1));
Nc = [length(find(label ==0)),length(find(label ==1))];
x(:,label ==1) = mvnrnd(mu1,sigma1,Nc(2))';
x(:,label ==0) = gmm_gen (n,Nc(1),alpha,mu0,sigma0);
subplot(2,2,i);
plot(x(1,label==0),x(2,label==0),'o',x(1,label==1),x(2,label==1),'x')        
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
%function g=eval_g(x,mu,Sigma)
%[n,N] = size(x);
%C = ((2*pi)^n * det(Sigma))^(-1/2);
%g = C*exp(E);
%end
function g = eval_g(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
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
s = alpha* g;

end

%------------
