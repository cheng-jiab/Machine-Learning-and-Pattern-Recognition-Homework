%Question2
clear all; close all;

N_train = 1000;
N_valid = 10000;
Num_L = 2;


[X_train, Label_train] = generateMultiringDataset(Num_L,N_train);
[X_valid, Label_valid] = generateMultiringDataset(Num_L,N_valid);

Label_train(Label_train==1) = -1;
Label_train(Label_train==2) = 1;
Label_valid(Label_valid==1) = -1;
Label_valid(Label_valid==2) = 1;

sigmalist = logspace(-1,1,40);
clist = logspace(0,2,40);
meanPE = zeros(length(clist),length(sigmalist));

K = 10;
dummy = ceil(linspace(0,N_train,K+1));
%for loop
for i = 1:length(clist)
    for j=1:length(sigmalist)
        [i,j]
        PE = zeros(1,K);
        parfor k = 1:K
            
            K_x_valid = X_train(:,[dummy(k)+1:dummy(k+1)]);
            K_y_valid = Label_train([dummy(k)+1:dummy(k+1)]);
            train_ind = setdiff([1:N_train],[dummy(k)+1:dummy(k+1)]);
            K_x_train = X_train(:,train_ind);
            K_y_train = Label_train(train_ind);
            %train svm
            SVMK = fitcsvm(K_x_train',K_y_train,'BoxConstraint',clist(i),'KernelFunction','gaussian','KernelScale',sigmalist(j));
            decisions = SVMK.predict(K_x_valid')';
            PE(k) = length(find(K_y_valid~=decisions))/ length(K_x_valid);

        end
    meanPE(i,j) = mean(PE)
    
    end
end

figure(2)
contour(log(clist),log(sigmalist),meanPE,20)
axis equal;
hold all;
plot(log(bestc),log(bestsigma),'ro')

[~,minind] = min(meanPE(:))
[indc,indsigma]=ind2sub(size(meanPE),minind);
bestc = clist(indc);
bestsigma = sigmalist(indsigma);

%final training
SVM_final = fitcsvm(X_train', Label_train','BoxConstraint',bestc,'KernelFunction','gaussian','KernelScale',bestsigma);
decisions_final = SVM_final.predict(X_valid')';
PEmin = length(find(Label_valid~=decisions_final))/ length(X_valid) 

%plot decisions
figure(3)
plot(X_valid(1,find(Label_valid==decisions_final)),X_valid(2,find(Label_valid==decisions_final)),'g+');
hold all;
plot(X_valid(1,find(Label_valid~=decisions_final)),X_valid(2,find(Label_valid~=decisions_final)),'ro');


function [data,labels] = generateMultiringDataset(numberOfClasses,numberOfSamples)

C = numberOfClasses;
N = numberOfSamples;
% Generates N samples from C ring-shaped 
% class-conditional pdfs with equal priors

% Randomly determine class labels for each sample
thr = linspace(0,1,C+1); % split [0,1] into C equal length intervals
u = rand(1,N); % generate N samples uniformly random in [0,1]
labels = zeros(1,N);
for l = 1:C
    ind_l = find(thr(l)<u & u<=thr(l+1));
    labels(ind_l) = repmat(l,1,length(ind_l));
end

a = [1:C].^2.5; b = repmat(1.7,1,C); % parameters of the Gamma pdf needed later
% Generate data from appropriate rings
% radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
angle = 2*pi*rand(1,N);
radius = zeros(1,N); % reserve space
for l = 1:C
    ind_l = find(labels==l);
    radius(ind_l) = gamrnd(a(l),b(l),1,length(ind_l));
end

data = [radius.*cos(angle);radius.*sin(angle)];

if 1
    colors = rand(C,3);
    figure(1), clf,
    for l = 1:C
        ind_l = find(labels==l);
        plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); axis equal, hold on,
    end
end
end
