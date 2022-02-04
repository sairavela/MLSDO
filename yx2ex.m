Xtrn = 0:0.2:1;
sXtrn = size(Xtrn);
noise = 0.0;
alg = [{'traingdx'},{'trainlm'},{'traingdm'},{'trainbr'}];
jj = 1;
for noise = 0:0.05:0.2
for algN = 1:1
for experiment = 1:20
    
    yTrn = Xtrn.^2+ noise*randn(sXtrn).*Xtrn;
    net = fitnet(1,char(alg(algN)));
    net = train(net,Xtrn,yTrn);
    
    xTst = rand/100:pi/100:2;
    yPred = net(xTst);
    
    figure(1);
        plot(Xtrn,yTrn,'ro'); hold on
        plot(xTst,[(xTst.^2)' yPred'],'LineWidth',2);
        plot([1 1], [0 4],'k','LineWidth',2);hold off;
        axis([0 2 0 4]);
        grid on
        title(sprintf('Performance: Algo: %s, Noise: %f',...
            char(alg(algN)),noise));
        legend('Training Data', 'Truth','Test','FontSize',16)

        drawnow;
        frame = getframe(gcf);
        im{jj} = frame2im(frame);
        jj = jj+1;
end
end
pause
end

%
filename = 'yx2ex.gif'; % Specify the output file name
 for idx = 1:jj-1  
     [A,map] = rgb2ind(im{idx},256);
     if idx == 1
         imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
     else
         imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.32);
     end
 end