clear all;
x = zeros(20,1);
figure(1); hold off;

poses = [6 9 12];

for i = 1:length(poses)
    pos = poses(i);
    tru(:,i) = x;
    tru(pos-1:pos+1,i) = 1;
end

%
for i= 1: 100
    pos = 6+ceil((rand>0.5)*6);
    y(:,i) = x+randn(size(x))*0.1;
    y(pos-1:pos+1,i) = 1+randn*0.1;
end
%
figure(1); plot(y)
%%
%load sparsedat;
w = 2*randn(size(tru,2),100);
%D = (y*w')*pinv(w*w');
D = tru;
e = (y-D*w);
J(1) = (0.5*trace(e'*e)+ 0.5*sum(sum(abs(w))))/100/20
dJ = -D'*e+sign(w); % Gradient L1
iter = 1;lr(1) = 0.001;
while ((J(end)/J(1)>0.01) && iter < 5000)
    iter  = iter + 1;
    lr(iter) = 0.001*log(iter);
    wn = w-lr(iter)*(pinv(D'*D))*dJ;
    e = (y-D*wn);
    J(iter) = (0.5*trace(e'*e)+ 0.5*sum(sum(abs(wn))))/100/20;
    dJ = -D'*e+ sign(wn);
    w = wn;
    %   D =((y*w')*pinv(w*w'));
    %
end
figure(2);
hold off;
subplot(221); plot(y,'LineWidth',2);
title('(a)  Sample Data ($y$)','FontSize',20,'interpreter','latex');
xlabel('Position Index','FontSize',18); ylabel('Amplitude','FontSize',18);
grid on;
yh = D*w;
subplot(224); plot(yh,'LineWidth',2);
title('(d) Estimate: $\hat{y}=B*\hat{w}_L$','interpreter','latex');
xlabel('Position Index','FontSize',18); ylabel('Amplitude','FontSize',18);
grid on;

cmap = colormap(gray(3))*0.75;
subplot(222);plot(D,'LineWidth',2);
title('(b) Bases ($B$)','interpreter','latex');set(gca,'ColorOrder',cmap);
legend('$\underbar{b}_1$','$\underbar{b}_2$','$\underbar{b}_3$','interpreter','latex');
grid on;
xlabel('Position Index','FontSize',18); ylabel('Amplitude','FontSize',18);
subplot(223);
yyaxis left;
plot(J(1:min(500,length(J))),'LineWidth',2);
title('(c) Cost ($J$)','interpreter','latex');


xlabel('Iterations','FontSize',18); ylabel('Cost','FontSize',18);
yyaxis right;
plot(lr(1:min(500,length(J))),'--','LineWidth',2);
ylabel('Learning Rate','FontSize',18);
grid on;
set(get(gcf,'children'),'FontSize',16);
set(gcf,'color','w');
saveas(gcf,'lassoperf.png');



%%
figure(3)

%
subplot(131); plot(w(1,:),w(2,:),'k.','MarkerSize',10);
title('(e) Subspace: ($\hat{w}_L[1], \hat{w}_L[2]$)','FontSize',20,'interpreter','latex');
xlabel('$\hat{w}_L[1]$','FontSize',18,'interpreter','latex');
ylabel('$\hat{w}_L[2]$','FontSize',18,'interpreter','latex');
ww=max(abs(w(:)));  axis([-ww ww -ww ww]);grid on
%
subplot(132); plot(w(1,:),w(3,:),'k.','MarkerSize',10);
axis([-ww ww -ww ww]);grid on
title('(f) Subspace: ($\hat{w}_L[1],\hat{w}_L[3]$)','FontSize',20,'interpreter','latex');
xlabel('$\hat{w}_L[1]$','FontSize',18,'interpreter','latex');
ylabel('$\hat{w}_L[3]$','FontSize',18,'interpreter','latex');
%
subplot(133); plot(w(2,:),w(3,:),'k.','MarkerSize',10);
title('(g) Subspace: ($\hat{w}_L[2],\hat{w}_L[3]$)','FontSize',20,'interpreter','latex');
xlabel('$\hat{w}_L[2]$','FontSize',18,'interpreter','latex');
ylabel('$\hat{w}_L[3]$','FontSize',18,'interpreter','latex');
axis([-ww ww -ww ww]);grid on

set(get(gcf,'children'),'FontSize',16);
set(gcf,'color','w');
drawnow;

saveas(gcf,'lassoweights.png');

%%

top = (y)'*(tru)/19;
bot = sqrt(diag(y'*y))*sqrt(diag(tru'*tru))';
rho = top./bot;
mutinf = -0.5*log(1-rho.^2);

%%
figure(4);
%
subplot(221); imagesc(mutinf./max(mutinf(:))); colorbar
title('(a) Mutual Information ($\cal{I}$)','FontSize',18,'interpreter','latex');
xlabel('Basis element','FontSize',18); ylabel('Sample Number','FontSize',18);
%
subplot(222)
bar(sum(mutinf)/sum(mutinf(:)),'LineWidth',2);
title('(b) Model Identification','FontSize',18,'interpreter','latex');
xlabel('Basis element','FontSize',18);
ylabel('Importance','FontSize',18);

[m,idx]=max(mutinf,[],2);
www = zeros(100,3);
for i = 1:length(idx)
    b = tru(:,idx(i));
    www(i,idx(i))=b'*y(:,i)/(b'*b);
end
www = www';
e = (y-D*www);
JDD = (0.5*trace(e'*e)+ 0.5*sum(sum(abs(www))))/100/20;

%
subplot(223);
plot(J(1:min(500,length(J))),'LineWidth',2);
title('(c) Cost ($J$)','interpreter','latex');
xlabel('Iterations','FontSize',18); ylabel('Cost','FontSize',18);
hold on;
plot([J(1) JDD*ones(1,499)],'k--','LineWidth',2);
hold off;
legend('$\ell_1$','DDDAS','interpreter','latex');
grid on;


%
subplot(224); plot(D*www,'LineWidth',2);
title('(d) Estimate: $\hat{y}_D=B*\hat{w}_D$','interpreter','latex');
xlabel('Position Index','FontSize',18); ylabel('Amplitude','FontSize',18);
grid on

set(get(gcf,'children'),'FontSize',16);
set(gcf,'color','w');
drawnow;
saveas(gcf,'dddasperf.png');
%%
figure(5)
%
subplot(131); plot(www(1,:),www(2,:),'k.','MarkerSize',10);
title('(e) Subspace: ($\hat{w}_D[1], \hat{w}_D[2]$)','FontSize',20,'interpreter','latex');
xlabel('$\hat{w}_D[1]$','FontSize',18,'interpreter','latex');
ylabel('$\hat{w}_D[2]$','FontSize',18,'interpreter','latex');
ww=max(abs(www(:)));  axis([-ww ww -ww ww]);grid on

%
subplot(132); plot(www(1,:),www(3,:),'k.','MarkerSize',10);
axis([-ww ww -ww ww]);grid on
title('(f) Subspace: ($\hat{w}_D[1],\hat{w}_D[3]$)','FontSize',20,'interpreter','latex');
xlabel('$\hat{w}_D[1]$','FontSize',18,'interpreter','latex');
ylabel('$\hat{w}_D[3]$','FontSize',18,'interpreter','latex');

%
subplot(133); plot(www(2,:),www(3,:),'k.','MarkerSize',10);
title('(g) Subspace: ($\hat{w}_D[2],\hat{w}_D[3]$)','FontSize',20,'interpreter','latex');
xlabel('$\hat{w}_D[2]$','FontSize',18,'interpreter','latex');
ylabel('$\hat{w}_D[3]$','FontSize',18,'interpreter','latex');
axis([-ww ww -ww ww]);grid on

set(get(gcf,'children'),'FontSize',16);
set(gcf,'color','w');
drawnow;
saveas(gcf,'dddasweights.png');