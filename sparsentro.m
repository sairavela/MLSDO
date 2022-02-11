
w_s = 0:0.01:1;

w = w_s.*randn(100,1)
for p = 0:0.5:2
plot(w_s,mean(abs(w).^p)/max(mean(abs(w).^p)),'LineWidth',2);hold on;
end
p = exp(-w.^2/2./w_s.^2)/sqrt(2*pi)./w_s;
ent = -sum(p.*log(p));
plot(w_s,(ent-min(ent))./(max(ent)-min(ent)),'LineWidth',2)
legend('0','0.5','1','1.5','2','ent');
axis([-0.1 1 0 1.1]);
hold off


