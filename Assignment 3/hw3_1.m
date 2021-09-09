clear all;
close all;
n=3;
C=4;
N_train = [100,200,500,1000,2000,5000];

P = repmat(1/C,1,C);
Mu = [1 1 0; 1 -1 0; -1 1 0; -1 -1 0]';
cov_factor = [0.4 0.5 0.5 0.7]; %factor to change overlap
for i = 1:C
Sigma(:,:,i) = cov_factor(i)*eye(3);
end
LossM = ones(C,C)-eye(C);

[x_o, label_o]  = generate_data(C,n,200000,P,Mu,Sigma,1); % generate data

ind_valid = ceil(200000*rand(1,100000)); %------valid
x_valid = x_o(:,ind_valid);
label_valid = label_o(:,ind_valid);

%theoretical p_error
for i = 1:C
    pxgiven(i,:) = eval_g(x_valid,Mu(:,i),Sigma(:,:,i));
    
end
px = P*pxgiven;
classPosteriors = pxgiven.*repmat(P',1,length(x_valid))./repmat(px,C,1);
expectedR = LossM*classPosteriors;
[~,decisions]=min(expectedR,[],1);

count = 0;
for i= 1:length(px)
    if  label_valid(decisions(i),i)==1
        count = count+1;
    end    
end
p_err_theo = 1-(count/length(px))



%for loop
Num_n = 6;%---------------
result = zeros(3,Num_n);
p_err_final = zeros(1,Num_n);
for E = 1:Num_n 
    %E=3
    
    tic
    
    ind = ceil(200000*rand(1,N_train(E))); %------train
    x = x_o(:,ind);
    label = label_o(:,ind);
    maxnP = 15 %---------------------
    p_err_nP = zeros(1,maxnP);
    for nP = 1:maxnP   
        
        nPerceptrons = nP;
        sizeParams = [n;nPerceptrons;C];
        %kfold
        K=10;%----------------
        dummy = ceil(linspace(0,N_train(E),K+1));
        p_err_K = zeros(1,K);
        parfor k = 1:K %enable parrallel
            K_valid = x(:,[dummy(k)+1:dummy(k+1)]);
            label_valid_K = label(:,[dummy(k)+1:dummy(k+1)]);
            train_ind = setdiff([1:N_train(E)],[dummy(k)+1:dummy(k+1)]);
            K_train = x(:,train_ind);
            label_K = label(:,train_ind);
        % Initialize model parameters
            paramsA = 1e-1*randn(nPerceptrons,n);
            paramsb = 1e-1*randn(nPerceptrons,1);
            paramsC = 1e-1*randn(C,nPerceptrons);
            paramsd = mean(label_K,2);%zeros(nY,1); % initialize to mean of y
            %params = paramsTrue;
            vecParamsInit = [paramsA(:);paramsb;paramsC(:);paramsd];
            %vecParamsInit = vecParamsTrue; % Override init weights with true weights

            options = optimset('MaxFunEvals',1e5*length(vecParamsInit)); % Matlab default is 200*length(vecParamsInit)

            % MaxFunEvals is the maximum number of function evaluations you allow

            vecParams = fminsearch(@(vecParams)(objectiveFunction(K_train,label_K,sizeParams,vecParams)),vecParamsInit,options);

            %validate
            paramsA = reshape(vecParams(1:n*nPerceptrons),nPerceptrons,n);
            paramsb = vecParams(n*nPerceptrons+1:(n+1)*nPerceptrons);
            paramsC = reshape(vecParams((n+1)*nPerceptrons+1:(n+1+C)*nPerceptrons),C,nPerceptrons);
            paramsd = vecParams((n+1+C)*nPerceptrons+1:(n+1+C)*nPerceptrons+C);
            H = mlpModel(K_valid,paramsA,paramsb,paramsC,paramsd);

            count = 0;
            for i= 1:length(H)
            [~,class] = max(int8(H(:,i)));
            [~,Label] = max(label_valid_K(:,i));
                if  class == Label
                    count = count+1;
                end    
            end
            p_err_K(k) = 1-(count/length(H));
        end
        [E,nP] %PROGRESS
        p_err_nP(nP) = min(p_err_K)
        figure(2)
        subplot(3,2,E)
        stem(nP,p_err_nP(nP)),hold on, drawnow,
        
    end
    [p_err,bestnP] =min(p_err_nP);
    result(1,E) = bestnP;
    result(2,E) = p_err;
    
    % final training
    %for loop
    Num_iterate = 5 %---------------
    p_err_f = zeros(1,Num_iterate);
    for M = 1:Num_iterate
        nPerceptrons = bestnP;
        sizeParams = [n;nPerceptrons;C];
        paramsA = 1e-1*randn(nPerceptrons,n);
        paramsb = 1e-1*randn(nPerceptrons,1);
        paramsC = 1e-1*randn(C,nPerceptrons);
        paramsd = mean(label,2);%zeros(nY,1); % initialize to mean of y
        %params = paramsTrue;
        vecParamsInit = [paramsA(:);paramsb;paramsC(:);paramsd];
        %vecParamsInit = vecParamsTrue; % Override init weights with true weights

        options = optimset('MaxFunEvals',1e6*length(vecParamsInit)); 

        vecParams = fminsearch(@(vecParams)(objectiveFunction(x,label,sizeParams,vecParams)),vecParamsInit,options);

        %validate
        paramsA = reshape(vecParams(1:n*nPerceptrons),nPerceptrons,n);
        paramsb = vecParams(n*nPerceptrons+1:(n+1)*nPerceptrons);
        paramsC = reshape(vecParams((n+1)*nPerceptrons+1:(n+1+C)*nPerceptrons),C,nPerceptrons);
        paramsd = vecParams((n+1+C)*nPerceptrons+1:(n+1+C)*nPerceptrons+C);
        H = mlpModel(x_valid,paramsA,paramsb,paramsC,paramsd);

        count = 0;
        for i= 1:length(H)
        [~,class] = max(int8(H(:,i)));
        [~,Label] = max(label_valid(:,i));
            if  class == Label
                count = count+1;
            end    
        end
        p_err_f(M) = 1-(count/length(H))
    end    
    p_err_final(E)= min(p_err_f);
    result(3,E) = p_err_final(E)
    toc
    figure(3)% size vs np
    stem(E,bestnP,'LineWidth',2),hold on, drawnow,
    xlim([0 7]);
    title('Best number of Perceptron vs. Number of Samples');

    figure(4) % size vs pe
    stem(E,p_err_final(E),'LineWidth',2),hold on, drawnow,
    xlim([0 7]);
    title('Minimum Prob. Error vs. Number of Samples');
end




function Value = objectiveFunction(X,Y,sizeParams,vecParams)
N = size(X,2); % number of samples
nX = sizeParams(1);
nPerceptrons = sizeParams(2);
nY = sizeParams(3);
paramsA = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
paramsb = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
paramsC = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
paramsd = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,paramsA,paramsb,paramsC,paramsd);
% Change the objective function appropriately
%objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N; % MSE for regression under AWGN model
Value = sum(-sum(Y.*log(H),1),2)/N; % CrossEntropy for ClassPosterior approximation
end

function H = mlpModel(X,paramsA,paramsb,paramsC,paramsd)
N = size(X,2);                          % number of samples
nY = length(paramsd);                  % number of outputs
U = paramsA*X + repmat(paramsb,1,N);  
Z = U./sqrt(1+U.^2);     %activation function
V = paramsC*Z + repmat(paramsd,1,N);  

H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer


end

function [x, label] = generate_data(C,n, N, P, Mu, Sigma,i)
x = zeros(n,N);
label = zeros(C,N);

cumP= [0 cumsum(P)];
u = rand(1,N);
for i=1:length(P)
    ind = find(u>=cumP(i) & u <cumP(i+1));
    label(i,ind)=1;
    x(:,ind) = mvnrnd(Mu(:,i),Sigma(:,:,i),length(ind))';
end
%subplot(3,2,i);
plot(x(1,label(1,:)==1),x(2,label(1,:)==1),'o',x(1,label(2,:)==1),x(2,label(2,:)==1),'x',x(1,label(3,:)==1),x(2,label(3,:)==1),'+',x(1,label(4,:)==1),x(2,label(4,:)==1),'*');
    
end

function g = eval_g(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end