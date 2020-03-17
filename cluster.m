
close all;clear all;clc
load arm_state.mat

%% Problem 1
figure; hold on 
subplot(2,1,1)
hold on
[coeff,score,latent,tsquared,explained,mu] = pca(Xplan);
plot(coeff(:,1),'r');plot(coeff(:,2),'g');plot(coeff(:,3),'b')
legend('PC1', 'PC2','PC3')
ylabel('Magnitude');xlabel('ms'); title('First three PC')

subplot(2,1,2)
plot(sqrt(latent),'o')
xlabel('Component #');ylabel('sqrt(eigenvalues'); 
title('sqrt of eigan values')
explained_variance=sum(explained(1:3))


%% 1B
figure
projection=Xplan*coeff(:,1:3);
color={'r.';'g.';'b.';'k.';'y.';'m.';'c.';'k.';'b.'};
for i=1:8
    pc1=projection((i-1)*91+1:91*i,1); pc2=projection((i-1)*91+1:91*i,2);pc3=projection((i-1)*91+1:91*i,3);
    plot3(pc1,pc2,pc3,color{i})
    hold on
end
xlabel('PC1');ylabel('PC2');
title('PC1 vs PC2 vs PC3'); zlabel('PC3')

%% 1C
figure
Um=coeff(:,1:3);
imagesc(Um)
colorbar

%% 1D
[lambda,psi,T,stats,F] = factoran(Xplan,3);
f_projection=Xplan*lambda;
figure
for i=1:8
    pc1=f_projection((i-1)*91+1:91*i,1); pc2=f_projection((i-1)*91+1:91*i,2);pc3=f_projection((i-1)*91+1:91*i,3);
    plot3(pc1,pc2,pc3,color{i})
    hold on
end
xlabel('F1');ylabel('F2');
title('Factor analysis'); zlabel('F3')


%% 2
% Initializing dataset
K=8;D=97;N=91;
train=zeros(D,200,N,K);
test=train;
for k=1:K
    for n=1:N
        train(:,:,n,k)=train_trial(n,k).spikes(:,351:550);
        test(:,:,n,k)=test_trial(n,k).spikes(:,351:550);
    end
end

train=squeeze(sum(train,2));
test=squeeze(sum(test,2));
class_prob=(1/K)*ones(1,K);

%% Mean and Covariance
mu=squeeze(sum(train,2))/N;

sigma=zeros(D,D);
for k=1:K
    for n=1:N
        x=train(:,n,k);
        sigma=sigma+(x-mu(:,k))*(x-mu(:,k))';
    end
end
sigma=sigma/91;

sigma_class=zeros(D,D,K);
for k=1:K
    for n=1:N
        x=train(:,n,k);
        sigma_class(:,:,k)=sigma_class(:,:,k)+(x-mu(:,k))*(x-mu(:,k))';
    end
end
sigma_class=sigma_class/N;



%% Full Gaussian Shared Covariance

% Testing
correct=0;incorrect=0;
for n=1:N
    for k=1:K
        x=test(:,n,k);
        prob_vect=(mu'*inv(sigma)*x)-diag((.5*mu'*inv(sigma)*mu))+log(class_prob');
        [v,l]=max(prob_vect);
        if l==k
            correct=correct+1;
        else
            incorrect=incorrect+1;
        end
    end
end
accuracy=correct/(correct+incorrect)

%% Full Gaussian Class covariance

% Testing
placement=zeros(1,8);
correct=0;incorrect=0;
for n=1:N
    for k=1:K
        x=test(:,n,k);
        prob_vect=log(class_prob);
        for t=1:K
            %prob_vect(t)=(mu(:,t)'*inv(sigma_class(:,:,t))*x)-(.5*mu(:,t)'*inv(sigma_class(:,:,t))*mu(:,t))-0.5*log(det(sigma_class(:,:,t)))+log(class_prob(t));
            diff=x-mu(:,t);
            prob_vect(t)=-0.5*(diff'*inv(sigma_class(:,:,t))*diff+log(det(sigma_class(:,:,t))));
        end
        [v,l]=max(prob_vect);
        placement(l)=placement(l)+1;
        if l==k
            correct=correct+1;
        else
            incorrect=incorrect+1;
        end
    end
end
accuracy_class=correct/(correct+incorrect)
% Warning is that the covariance is close to singular or singular
%% Diagonal Gaussian, shared covariance

correct=0;incorrect=0;
sigma_diag=diag(sigma);
for n=1:N
    for k=1:K
        prob_vect=ones(1,8)*log(class_prob(1));
        for t=1:K
            for i=1:D
                x=test(i,n,k);
                prob_vect(t)=prob_vect(t)+(mu(i,t)*x/sigma_diag(i))-(0.5*mu(i,t)^2/sigma_diag(i));
            end
        end
    [v,l]=max(prob_vect);
    if l==k
        correct=correct+1;
    else
        incorrect=incorrect+1;
    end
    end
end
accuracy_diag=correct/(correct+incorrect)

%% Diagonal Gaussian, class covariance

correct=0;incorrect=0;

for n=1:N
    for k=1:K
        prob_vect=ones(1,8)*log(class_prob(1));
        for t=1:K
            sigma_diag_class=diag(sigma_class(:,:,t));
            for i=1:D
                x=test(i,n,k);
%               prob_vect(t)=prob_vect(t)+(mu(i,t)*x*inv(sigma_diag_class(i)))-(0.5*mu(i,t)^2*inv(sigma_diag_class(i)));
                diff=x-mu(i,t);
                prob_vect(t)=prob_vect(t)-0.5*((diff'*inv(sigma_diag_class(i))*diff)+log(sigma_diag_class(i)));
            end
        end
    [v,l]=max(prob_vect);
    if l==k
        correct=correct+1;
    else
        incorrect=incorrect+1;
    end
    end
end
accuracy_diag_class=correct/(correct+incorrect)


%% Minimum variance for diagonal gaussian class covariance

correct=0;incorrect=0;

sigma_class_aug=sigma_class;
sigma_class_aug(sigma_class_aug<0.01)=0.01;

for n=1:N
    for k=1:K
        prob_vect=ones(1,8)*log(class_prob(1));
        for t=1:K
                sigma_diag_class=diag(sigma_class_aug(:,:,t));
            for i=1:D
                x=test(i,n,k);
%               prob_vect(t)=prob_vect(t)+(mu(i,t)*x*inv(sigma_diag_class(i)))-(0.5*mu(i,t)^2*inv(sigma_diag_class(i)));
                diff=x-mu(i,t);
                prob_vect(t)=prob_vect(t)-0.5*(diff'*inv(sigma_diag_class(i))*diff-log(sigma_diag_class(i)));
            end
        end
    [v,l]=max(prob_vect);
    if l==k
        correct=correct+1;
    else
        incorrect=incorrect+1;
    end
    end
end
accuracy_diag_class_aug=correct/(correct+incorrect)

%% Poisson model

mu_poisson=mu;
mu_poisson(mu_poisson<0.01)=0.01;
placement=zeros(1,K);

correct=0; incorrect=0;
for n=1:N
    for k=1:K
        prob_vect=ones(1,8)*log(class_prob(1));
        for t=1:K
            for i=1:D
                x=test(i,n,k);
                prob_vect(t)=prob_vect(t)-mu_poisson(i,t)+x*log(mu_poisson(i,t));
            end
        end
    [v,l]=max(prob_vect);
    placement(l)=placement(l)+1;
    if l==k
        correct=correct+1;
    else
        incorrect=incorrect+1;
    end
    end
end
accuracy_poisson=correct/(correct+incorrect)

