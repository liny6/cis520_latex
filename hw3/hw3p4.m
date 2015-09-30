load 'hw3-data/data.mat'
lambda = 1;
%f = @(w)norm(Y-X*w, 2) + lambda*norm(w, 1);

%[w, fval] = fminsearch(f, [0;0;0]);

f1 = @(w)norm(Y-X*[0; 0; 0], 2) + 0;
f2 = @(w)norm(Y-X*[w(1); 0; 0], 2) + 1;
f3 = @(w)norm(Y-X*[0; w(2); 0], 2) + 1;
f4 = @(w)norm(Y-X*[0; 0; w(3)], 2) + 1;
f5 = @(w)norm(Y-X*[w(1); w(2); 0], 2) + 2;
f6 = @(w)norm(Y-X*[w(1); 0; w(3)], 2) + 2;
f7 = @(w)norm(Y-X*[0; w(2); w(3)], 2) + 2;
f8 = @(w)norm(Y-X*w, 2) + 3;

f_MLE = @(w)norm(Y-X*w, 2);
[w_MLE, fval] = fminsearch(f_MLE, [0;0;0]);
ratio6a = norm(w_MLE,2)/norm(Y-X*w_MLE,2);

lambda = rand;
f = @(w)norm(Y-X*w, 2) + lambda*norm(w, 2);

w = fminsearch(f, [0;0;0]);
ratio6cd = norm(w,2)/norm(w_MLE,2);

while(ratio6cd < 0.4 || ratio6cd >0.5)
    lambda = rand*10;
    f = @(w)norm(Y-X*w, 2) + lambda*norm(w, 2);
    w = fminsearch(f, [0;0;0]);
    ratio6cd = norm(w,2)/norm(w_MLE,2);
end
    
