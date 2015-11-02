%% Plots/submission for SVM portion, Question 1.

%% Put your written answers here.
clear 
answers{1} = 'The intersection kernel gives me the least training error. This makes sense because from a language perspective, the intersection kernel is a better measure of how close the two documetns are compared to the others.';

save('problem_1_answers.mat', 'answers');

%% Load and process the data.
addpath ./libsvm
load ../data/windows_vs_mac.mat;
[X, Y] = make_sparse(traindata, vocab);
[Xtest, Ytest] = make_sparse(testdata, vocab);

%% Bar Plot - comparing error rates of different kernels

% INSTRUCTIONS: Use the KERNEL_LIBSVM function to evaluate each of the
% kernels you mentioned. Then run the line below to save the results to a
% .mat file.
kernel_l = @(x, x2) kernel_poly(x, x2, 1);
kernel_q = @(x, x2) kernel_poly(x, x2, 2);
kernel_c = @(x, x2) kernel_poly(x, x2, 3);
kernel_g20 = @(x, x2) kernel_gaussian(x, x2, 20);
kernel_i = @(x, x2) kernel_intersection(x, x2);

[test_err_l, info_l] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_l);
[test_err_q, info_q] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_q);
[test_err_c, info_c] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_c);
[test_err_g20, info_g20] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_g20);
[test_err_i, info_i] = kernel_libsvm(X, Y, Xtest, Ytest, kernel_i);


results.linear = test_err_l;% ERROR RATE OF LINEAR KERNEL GOES HERE
results.quadratic = test_err_q;% ERROR RATE OF QUADRATIC KERNEL GOES HERE
results.cubic = test_err_c;% ERROR RATE OF CUBIC KERNEL GOES HERE
results.gaussian = test_err_g20;% ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
results.intersect = test_err_i;% ERROR RATE OF INTERSECTION KERNEL GOES HERE

%%

% Makes a bar chart showing the errors of the different algorithms.
algs = fieldnames(results);
for i = 1:numel(algs)
    y(i) = results.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Kernel');
ylabel('Test Error');
title('Kernel Comparisons');

%print -djpeg -r72 plot_1.jpg;
