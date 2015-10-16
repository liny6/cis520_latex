%% data and shit
X = [0 -1;
     1, 0;
     -1,0;
     0, 1];
X1 = X(:, 1);
X2 = X(:, 2);
 
[xnr, xnc] = size(X);

features = 2;
splits = 2;

X_split = [-0.5, 0.5];

Y = [-1; 1; 1; -1];

p_Y1 = sum(Y==1)/length(Y);

H_Y = binary_entropy(p_Y1);

T = 4;
%initial D_t(i)
D = ones(xnr, xnr + 1)/xnr;

h = zeros(xnr, T);

alpha = zeros(1, T);

episolon = zeros(1, T);
Z = zeros(1, T);


for t = 1:T
    
    %% train basic decision tree stump
    IG_best = 0;
    best_split = 0;
    best_feature = 0;
    
    for i = 1:features
        for j = 1:splits
            ind = bsxfun(@le, X(:,i), X_split(j));
            p_X1 = sum(ind)/length(ind);
            Y_givenX1 = bsxfun(@times, Y, ind);
            Y_givenX0 = bsxfun(@times, Y, ~ind);
            p_Y1_givenX1 = sum(D(:, t).*(Y_givenX1 == 1))/sum(D(:, t).*ind);
            p_Y1_givenX0 = sum(D(:, t).*(Y_givenX0 == 1))/sum(D(:, t).*(~ind));
            H_Y1_givenX = p_X1*(binary_entropy(p_Y1_givenX1)) + (1 - p_X1)*binary_entropy(p_Y1_givenX0);
            IG = H_Y - H_Y1_givenX;
            if IG > IG_best
                IG_best = IG;
                best_split = j;
                best_feature = i;
            end
        end
    end
    
    %% predict dataset with the stump
    indh = bsxfun(@le, X(:, best_feature), X_split(best_split));
    
    %find majority
    left_weight = sum(D(:, t).*indh.*Y);
    %right_weight = sum(D.*~indh.*Y);
    
    if left_weight < 0
        h(:, t) = (bsxfun(@ge, X(:,best_feature), X_split(best_split))*2)-1;
    else
        h(:, t) = (bsxfun(@le, X(:,best_feature), X_split(best_split))*2)-1;
    end
   
     %% Choose alpha
    episolon(t) = sum(D(:, t).*(bsxfun(@ne, Y, h(:, t))));
    alpha(t) = 1/2*log((1-episolon(t))/episolon(t));
    
    %% update D and Z
    Z(t) = sum(D(:, t).* bsxfun(@power, 2.71828, -alpha(t).*Y.*h(:,t)));
    D(:, t + 1) = D(:, t).*bsxfun(@power, 2.71828, -alpha(t).*Y.*h(:, t)) / Z(t);   
    
    figure(t)
    hold on
    for i = 1:T
        if Y(i) == 1
            plot(X1(i), X2(i), 'b+', 'markersize', 50*D(i, t));
        else
            plot(X1(i), X2(i), 'b*', 'markersize', 50*D(i, t));
        end
    end
    
    if best_feature == 1
        plot([X_split(best_split), X_split(best_split)], [-1.5 , 1.5])
    else
        plot([-1.5, 1.5], [X_split(best_split), X_split(best_split)])
    end
    
    titletext = sprintf('Step %d', t);
    title(titletext)
    
    axis([-1.5, 1.5, -1.5, 1.5])
end

%% take the weighted sum of the stumps to make the final decision
    Y_hat = sum(bsxfun(@times, h(:,1:t), alpha(1:t)), 2);