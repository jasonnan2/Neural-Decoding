
clear all;close all;clc
load move.mat

%% Problem 1
close all
N=91;K=8;D=97;bin=20;
% Processing training data
T_min=ones(1,K)*100;
T=zeros(N,K);
for n=1:N
    for k=1:K
        train_data=train_trial(n,k).spikes;
        train_arm=train_trial(n,k).handPos;
        T(n,k)=floor(size(train_data,2)/bin);
        if T(n,k)<T_min(k)
            T_min(k)=T(n,k);
        end
        for t=1:T(n,k)
            range=train_data(:,(t-1)*20+1:20*t);
            train(n,k).spikes(:,t)=sum(range,2);
            train(n,k).arm(1,t)=train_arm(1,20*t);
            train(n,k).arm(2,t)=train_arm(2,20*t);
            if t==1
                train(n,k).arm(3,t)=(train_arm(1,20)-train_arm(1,1))/20;
                train(n,k).arm(4,t)=(train_arm(2,20)-train_arm(2,1))/20;
            else 
                %train(n,k).arm(3,t)=(train_arm(1,20*t)-train_arm(1,(t-1)*20+1))/20;
                %train(n,k).arm(4,t)=(train_arm(2,20*t)-train_arm(2,(t-1)*20+1))/20;
                train(n,k).arm(3,t)=(train(n,k).arm(1,t)-train(n,k).arm(1,t-1))/20;
                train(n,k).arm(4,t)=(train(n,k).arm(2,t)-train(n,k).arm(2,t-1))/20;
            end
        end
        train(n,k).data=sum(train(n,k).spikes);
    end
end

%% Calculating mu
figure
hold on
for k=1:8
    mean=zeros(1,T_min(k));
    for n=1:N
        data=train(n,k).spikes(:,1:T_min(k));
        mean=mean+sum(data);
    end
    mu(k).spikes=mean/N;
    plot(mu(k).spikes)
end
title('Mean Spike count of all neurons across time')
ylabel('Spike Count'); xlabel('Time (ms)')
set(gca,'Xticklabel',linspace(20,max(T_min)*20,max(T_min)))
legend({'K=1';'K=2';'K=3';'K=4';'K=5';'K=6';'K=7';'K=8'},'location','best')

%% plotting trace of arm
figure
hold on
counter=1;
for k=1:K
    selected=ceil(rand(1,10)*N);
    while length(unique(selected))<10
        selected=ceil(rand(1,10)*N);
    end
    subplot(4,4,counter)
    hold on
    title(['Position of arm K=',int2str(k)])
    set(gca,'Xticklabel',linspace(20,T_min(k)*20,T_min(k)))
    counter=counter+1;
    % plot horizontal and vertical position
    for n=selected
        Px=train(n,k).arm(1,1:T_min(k));
        Py=train(n,k).arm(2,1:T_min(k));
        plot(Px,'k')
        plot(Py,'b')
    end
    legend({'X';'Y'},'location','best')
    hold off
    % Plotting velocity
    subplot(4,4,counter)
    hold on
    title(['velocity of arm K=',int2str(k)])
    set(gca,'Xticklabel',linspace(20,T_min(k)*20,T_min(k)))
    counter=counter+1;
    % plot horizontal and vertical position
    for n=selected
        Vx=train(n,k).arm(3,1:T_min(k));
        Vy=train(n,k).arm(4,1:T_min(k));
        plot(Vx,'r')
        plot(Vy,'g')
    end
    legend({'Vx';'Vy'},'location','best')
    hold off
end

%% Problem 2: Defining parameters
T_max=max(T);
T_sum_sum=sum(sum(T));
T_sum=sum(T);
% FIX NORMALIZATION
Parameters.A=zeros(4,4);
top=zeros(4,4);
bottom=top;
for k=1:K
    for n=1:N
        %top=zeros(4,4);
        %bottom=top;
        for t=2:T(n,k)
            zt=train(n,k).arm(:,t);
            zt1=train(n,k).arm(:,t-1);
            top=top+zt*zt1';
            bottom=bottom+zt1*zt1';
        end
        %Parameters.A=Parameters.A+((top*inv(bottom))*T(n,k)/T_sum_sum);
    end
end
Parameters.A=Parameters.A+top*inv(bottom);
Parameters.A=Parameters.A;

%Parameters.A=[1 0 20 0;0 1 0 20;0 0 1 0; 0 0 0 1];

Parameters.Q=zeros(4,4);

for k=1:8
    for n=1:N
        holder=zeros(4,4);
        for t=2:T(n,k)
            zt=train(n,k).arm(:,t);
            zt1=train(n,k).arm(:,t-1);
            holder=holder+(zt-Parameters.A*zt1)*(zt-Parameters.A*zt1)';
        end
        Parameters.Q=Parameters.Q+inv(T(n,k)-1)*holder*T(n,k)/T_sum_sum;
    end
end
Parameters.Q=Parameters.Q;
Parameters.Q=[0 0 0 0;0 0 0 0; 0 0 0.0162 -0.007;0 0 -0.007 0.0136];
A=Parameters.A
Q=Parameters.Q

C=zeros(97,4);
holder=zeros(97,4);
top=holder;
bottom=zeros(4,4);

for k=1:K
    for n=1:N
        for t=1:T(n,k)
            xt=train(n,k).spikes(:,t);
            zt=train(n,k).arm(:,t);
            top=top+xt*zt';
            bottom=bottom+zt*zt';
        end
        % Normalize?
        
    end
end
C=C+(top*inv(bottom));
%C=C/(N*K)
Parameters.C=C;

R=zeros(D,D);
for k=1:K
    for n=1:N
        holder=zeros(97,97);
        for t=1:T(n,k)
            xt=train(n,k).spikes(:,t);
            zt=train(n,k).arm(:,t);
            holder=holder+(xt-C*zt)*(xt-C*zt)';
        end
        R=R+inv(T(n,k))*holder*T(n,k)/T_sum_sum;
    end
end

%R=R/(N*K)
Parameters.R=R;

init=[];
for k=1:K
    for n=1:N
    init=[init,train(n,k).arm(:,1)];
    end
end
clear mean
Zi=mean(init,2);
V=cov(init');

%% Testing Phase

%% Processing testing data
T_min_test=ones(1,K)*100;
T_test=zeros(N,K);
for n=1:N
    for k=1:K
        test_data=test_trial(n,k).spikes;
        test_arm=test_trial(n,k).handPos;
        T_test(n,k)=floor(size(test_data,2)/bin);
        if T_test(n,k)<T_min_test(k)
            T_min_test(k)=T_test(n,k);
        end
        for t=1:T_test(n,k)
            range=test_data(:,(t-1)*20+1:20*t);
            test(n,k).spikes(:,t)=sum(range,2);
            test(n,k).arm(1,t)=test_arm(1,20*t);
            test(n,k).arm(2,t)=test_arm(2,20*t);
            if t==1
                test(n,k).arm(3,t)=(test_arm(1,20)-test_arm(1,1))/20;
                test(n,k).arm(4,t)=(test_arm(2,20)-test_arm(2,1))/20;
            else 
                %train(n,k).arm(3,t)=(train_arm(1,20*t)-train_arm(1,(t-1)*20+1))/20;
                %train(n,k).arm(4,t)=(train_arm(2,20*t)-train_arm(2,(t-1)*20+1))/20;
                test(n,k).arm(3,t)=(test(n,k).arm(1,t)-test(n,k).arm(1,t-1))/20;
                test(n,k).arm(4,t)=(test(n,k).arm(2,t)-test(n,k).arm(2,t-1))/20;
            end
        end
        test(n,k).data=sum(test(n,k).spikes);
    end
end

%% Testing 

for k=1:K
    for n=1:N
        mu_update=zeros(4,T_test(n,k));
        mu_update(:,1)=Zi;
        sigma_update=zeros(4,4,T_test(n,k));
        sigma_update(:,:,1)=(V);
        prediction(n,k).state(:,1)=Zi;
        prediction(n,k).cov(:,:,1)=V;
        for t=2:T_test(n,k)
            x=test(n,k).spikes(:,t);
            
            %One step prediction
            mu_update(:,t)=A*mu_update(:,t-1);
            sigma_update(:,:,t)=A*sigma_update(:,:,t-1)*A'+Q;
            sig=sigma_update(:,:,t);
            
            % Measurement update
            kalman=sig*C'*inv(C*sig*C'+R);
            guess=mu_update(:,t)+kalman*(x-C*mu_update(:,t));
            prediction(n,k).state(:,t)=guess;
            mu_update(:,t)=guess;
            covariance=sig-kalman*C*sig;
            prediction(n,k).cov(:,:,t)=covariance;
            sigma_update(:,:,t)=covariance;
        end
    end
end

%% Plotting all testing

figure
hold on

for k=1:K
    subplot(4,2,k)
    title(['K=',int2str(k)])
    hold on
    for n=1:N
        x=test(n,k).arm(1,:);
        y=test(n,k).arm(2,:);
        plot(x,y,'c')
        predict_x=prediction(n,k).state(1,:);
        predict_y=prediction(n,k).state(2,:);
        plot(predict_x,predict_y,'r')
    end
end

%sgtitle('Crack Head Monkey Steve')
        
%% Plotting trial 1,4

figure; hold on
subplot(2,1,1)
hold on
x=test(1,1).arm(1,:);
y=test(1,1).arm(2,:);
real=plot(x,y,'k');
predict_x=prediction(1,1).state(1,:);
predict_y=prediction(1,1).state(2,:);
predict_plot=plot(predict_x,predict_y,'r');
for t=1:length(predict_x)
    cov_hold=prediction(1,1).cov(:,:,t);
    plot_gaussian_ellipsoid([predict_x(t) predict_y(t)],[cov_hold(1:2,1:2)])
end
title('Arm State 1,1');xlabel('Xmm');ylabel('Ymm')
legend([real,predict_plot],'true','predicted','location','best');
hold off

subplot(2,1,2)
hold on
x=test(1,4).arm(1,:);
y=test(1,4).arm(2,:);
real=plot(x,y,'k');
predict_x=prediction(1,4).state(1,:);
predict_y=prediction(1,4).state(2,:);
predict_plot=plot(predict_x,predict_y,'r');
for t=1:length(predict_x)
    cov_hold=prediction(1,4).cov(:,:,t);
    plot_gaussian_ellipsoid([predict_x(t) predict_y(t)],[cov_hold(1:2,1:2)])
end
title('Arm State 1,4');xlabel('Xmm');ylabel('Ymm')
legend([real,predict_plot],'true','predicted','location','best');
hold off
            
%% Error calculation
%% Error calculation
T_min_min=min(min(T));
avg_error_bin=zeros(1,T_min_min);
total_error=0;
for k=1:K
    for n=1:K
        x_real=test(n,k).arm(1,1:T_min_min);
        y_real=test(n,k).arm(2,1:T_min_min);
        x_predict=prediction(n,k).state(1,1:T_min_min);
        y_predict=prediction(n,k).state(2,1:T_min_min);
        avg_error_bin=avg_error_bin+sqrt((x_real-x_predict).^2+(y_real-y_predict).^2);
        % Total error
        x_real_t=test(n,k).arm(1,:);
        y_real_t=test(n,k).arm(2,:);
        x_predict_t=prediction(n,k).state(1,:);
        y_predict_t=prediction(n,k).state(2,:);
        total_error=total_error+sum(sqrt((x_real_t-x_predict_t).^2+sum(y_real_t-y_predict_t).^2));
    end
end
avg_error_bin=avg_error_bin/(N*K);
total_error=total_error/(n*k)


