q = 0:0.01:1;
p1 = 1-q;
p2 = (1-q).^2./(q.^2 + (1-q).^2);
q_bool = bsxfun(@ge, p1, p2);


figure
plot(p1, q, 'r')
hold on
plot(p2, q, 'b')
for i = 1:100
    dots = linspace(p1(i),p2(i), 10);
    plot(dots, linspace(q(i),q(i), 10), 'g.');
end

title('decision boundaries')
legend('true boundary', 'double counted X_2', 'area of error')
xlabel('p')
ylabel('q')

axis([0 1 0 1])