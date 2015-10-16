load 'data/train_y.mat'
load 'data/train_data.mat'

y_test_cell = load('data/test_y.mat');
data_test_cell = load('data/test_data.mat');

n = 64;

y_test = y_test_cell.y;
data_test = data_test_cell.data;

MLEf1 = @(w)norm(y - data(:,1)*w(1), 2)^2;
MLEf2 = @(w)norm(y - data(:,1:2)*w(1:2), 2)^2;
MLEf3 = @(w)norm(y - data*w, 2)^2;

[w1_MLE, err1_MLE] = fminsearch(MLEf1, [0;0;0]);
[w2_MLE, err2_MLE] = fminsearch(MLEf2, [0;0;0]);
[w3_MLE, err3_MLE] = fminsearch(MLEf3, [0;0;0]);

err1_MLE_bits = n*log2(err1_MLE/n);
err2_MLE_bits = n*log2(err2_MLE/n);
err3_MLE_bits = n*log2(err3_MLE/n);

AIC1_bits = err1_MLE_bits + 2;
AIC2_bits = err2_MLE_bits + 4;
AIC3_bits = err3_MLE_bits + 6;

BIC1_bits = err1_MLE_bits + 2*(0.5)*log2(n);
BIC2_bits = err2_MLE_bits + 2*(0.5)*log2(n);
BIC3_bits = err3_MLE_bits + 2*(0.5)*log2(n);

err1_MLEt = norm(y_test - data_test(:,1)*w1_MLE(1), 2).^2;
err2_MLEt = norm(y_test - data_test(:,1:2)*w2_MLE(1:2), 2).^2;
err3_MLEt = norm(y_test - data_test*w3_MLE).^2;

err1_MLE_bitst = n*log2(err1_MLEt/n);
err2_MLE_bitst = n*log2(err2_MLEt/n);
err3_MLE_bitst = n*log2(err3_MLEt/n);

AIC1_bitst = err1_MLE_bitst + 2;
AIC2_bitst = err2_MLE_bitst + 4;
AIC3_bitst = err3_MLE_bitst + 6;

BIC1_bitst = err1_MLE_bitst + 2*(0.5)*log2(n);
BIC2_bitst = err2_MLE_bitst + 2*(0.5)*log2(n);
BIC3_bitst = err3_MLE_bitst + 2*(0.5)*log2(n);
