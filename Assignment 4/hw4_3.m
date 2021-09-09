clear all; close all;

filenames ={'3096_color.jpg','42049_color.jpg'};
for i = 1:2
    data = imread(filenames{i});
    figure(1)
    subplot(2,1,i);
    imshow(data);
    [R,C,D] = size(data);
    N=R*C;
    data = double(data);;
    rowind = [1:R]'*ones(1,C);
    colind = ones(R,1)*[1:C];
    features = [rowind(:)';colind(:)'];
    
    for d = 1:D
        data_d = data(:,:,d);
        features = [features; data_d(:)'];
    end
    
    minf = min(features,[],2);
    maxf = max(features,[],2);
    ranges = maxf-minf;
    %norm
    x=(features-minf)./ranges;
   
    GMM2 = fitgmdist(x',2,'Replicates',10)
    post2 = posterior(GMM2,x')';
    decisions = post2(2,:)>post2(1,:);
    
    %plot gmm2
    label2 = reshape(decisions,R,C);
    figure(2)
    subplot(2,1,i);
    imshow(uint8(label2*255/2));
    
    %start k-fold
    M=10; %numebr of gmm
    K=10;
    N = length(x);
    dummy = ceil(linspace(0,N,K+1));
    for m = 1:M
        m
        parfor k = 1:K

            K_x_valid = x(:,[dummy(k)+1:dummy(k+1)]);
            train_ind = setdiff([1:N],[dummy(k)+1:dummy(k+1)]);
            K_x_train = x(:,train_ind);
            %fit
            GMMk = fitgmdist(K_x_train',m,'Replicates',5);
            if GMMk.Converged
                prob(k) = sum(log(pdf(GMMk,K_x_valid')));
            else
                prob(k) = 0
                
            end              
        end
    
    meanProb(i,m)= mean(prob)
    
    end
    
    %plot prob vs m
    figure(3)
    subplot(2,1,i)
    stem(meanProb(i,:))
    
    [~,bestM] = max(meanProb(i,:));
    resultM(i) = bestM;
    GMM_final = fitgmdist(x',bestM,'Replicates',10);
    postk = posterior(GMM_final,x')';
    
    lossMatrix = ones(bestM,bestM)-eye(bestM);
    
    expectedRisk = lossMatrix * postk;
    
   [~,decisions] = min(expectedRisk,[],1);
   
   %Plot
   labelk = reshape(decisions-1,R,C);
   figure(4)
   subplot(2,1,i)
   imshow(uint8(labelk*255/2));
   title(strcat({'Select best M = '},num2str(bestM)));
      
end

