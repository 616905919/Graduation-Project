x=[-10:0.1:20];
y=normpdf(x,0,1);%正态分布函数。
z=normpdf(x,2,1);
figure;
hold on;
plot(x,z,'--');
plot(x,y,'-');
legend('少数类','多数类');
plot([1,1],[0,10],'-');
plot([2,2],[0,10],'-');
plot([1.5,1.5],[0,10],'-');
text(1,0.4,'a');
text(2,0.35,'b');
text(1.5,0.3,'c');
hold off;
axis([-3, 8, 0, 0.5]);
%figure;
%axes1=axes('Pos',[0.1 0.1 0.8 0.8]);

%set(axes1,'YLim',[-0.01 0.43],'XLim',[-3 6]);