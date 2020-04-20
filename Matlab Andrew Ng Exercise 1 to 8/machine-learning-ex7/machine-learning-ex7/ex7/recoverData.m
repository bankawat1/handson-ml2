function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%        
%disp("K");disp(K); %5
disp("size(X_rec)");disp(size(X_rec));  %15 X 11     
%disp("size(Z)");disp(size(Z)) ; %15 X 5
%disp("size(U)");;disp(size(U)); %11 X 11
%tr = U';
%t1 = U'(:,1 : K);
%t = U'(:,K);
%disp("size(t1)");disp(size(t1));

%Z = X*U(:,1 : K);
X_rec  = (U(:, 1 : K)*Z')';

%X_rec_temp = t1*Z';


%X_rec = X_rec_temp';
% =============================================================

end
