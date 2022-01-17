clear all;
%Simulation parameters
nev = 1; nsen = 3; % Please don't try this with >1 event, the code needs to be modified.
ndim = 2;
%Random event
zt= rand(nev,ndim); %location and time
%sensor placement
s = [0 (rand-0.5)/4+0.5 1];
%velocity model
v = 0.5;
% random noise -- 1\%
sig = 0.01;
%arrival
y=abs(zt(:,1)-s)/v+zt(:,2)+randn(nev,nsen)*sig;
disp(y)

% Estimation
nens = nsen*nev*ndim/sqrt(sig); % This is a kludge

z = [rand(nev,ndim,nens)];
vens=rand(1,nens);
esprev = 0; des = inf;
jj = 1;
while(jj<100 && max(abs(des))>1e-4) % need better convergence criterion
    Z = [reshape(z,[nev*ndim nens]);vens];
    dZ = Z - mean(Z,2);
    
    Yh = ttModel(z,vens,s);
    dYh = Yh - mean(Yh,2);
    
    Czy = (dZ*dYh')/(nens -1 );
    Cyy  = cov(Yh');
    
    G = Czy*pinv(Cyy + sig^2*eye(nsen)); %Kalman Gain, inefficient approach.
    errv = (y' - Yh); % y wasn't perturbed.
    zup = G*errv;
    
    % use continued variance reduction as criterion
    es = var(zup,[],2);
    des = es - esprev;
    esprev = es;
    
    %uppdate
    Z = Z + zup;
    z = reshape(Z(1:2,:),nev,ndim,nens);
    vens = Z(3,:);
    %iterate
    jj = jj+1;
end
%answer
[mean(Z,2) [zt';v]]
disp(jj)



function y = ttModel(z,vens,s)
for i = 1:size(z,3)
    y(:,i) = abs(z(:,1,i)-s)./vens(i)+z(:,2,i);
    % y = abs(x - s)/v + t
    %% for the other approach
    % [sgn(x-s)/v        0        0; ...
    %  0          -abs(x-s)/v^2   0;...
    %  0                 0        1]
end
end
%%