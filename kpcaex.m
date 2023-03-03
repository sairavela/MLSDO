%% KPCA Example with a kernel
% Sai Ravela (C) 2017
% No explicit feature map calculation

close all;
set(0,'DefaultFigureWindowStyle','docked')
N2=100;
% Generate Data
th1 = rand(N2,1)*2*pi;
th2 = rand(N2,1)*2*pi;
amp1 = randn(N2,1)*0.1+2;
amp2 = randn(N2,1)*0.2+4;

x1=[amp1.*cos(th1) amp1.*sin(th1)];
x2=[amp2.*cos(th2) amp2.*sin(th2)];

data = [x1;x2];
[N,n]=size(data);

%Inner products
D=data*data';

%Kernel
K=D+diag(D)*diag(D)';

%Centering
tmp = (1/N*ones(1,N)*K + K*ones(N,1)/N);
tmp = (tmp + tmp')/2;
K = K - tmp+ ones(1,N)*K*ones(N,1)/N^2;

[u,s,v]=svd(K);

%Embedding
z=1/sqrt(N)*pinv(sqrt(s))*u'*K;

clrs = [repmat([1 0 0],[100 1]); repmat([0 0 1],[100 1])];
h1=figure(1);
scatter(data(:,1),data(:,2),10,clrs,'filled')
figure(2); plot(ksdensity(z(1,:)),'r.');

figure(4);scatter(z(1,:),z(2,:),10,clrs,'filled');

