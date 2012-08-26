# the program assumes normal (gaussian) distribution of all the features in the dataset and builds a decision boundary based classifier to detect spam emails #

# Author:
#	Pranav Gupta 
#	Department of Computer Science and Engineering
#	Indian Institute of Technology Guwahati

#discriminant function in case of normal distribution :
#	g-i(x-bar) = [-1/2 * (x-bar - u-i-bar)' * sigma-i-inverse * (x-bar - u-i-bar)]  - d/2 ln(2*pi)  - 1/2|sigma-i| + ln(P(w-i))
#
#	Assuming the independence of features, we have
#		g-i(x-bar) = [-1/2 * (x-bar - u-i-bar)' * sigma-i-inverse * (x-bar - u-i-bar)]  + ln(P(w-i))
#		
#	so, required quantities to be calculated from the data:
#		1. u-1, sigma-1, u-2, sigma-2
#		2. |sigma-1|, |sigma-2|

# Fetch training data into a Matrix
M = csvread('spambase.data.txt');

# divide the data of different classes
# ====================================
W1 = zeros(0,57);
W2 = zeros(0,57);

for rowno = [1:rows(M)]
	rowM = M(rowno,:);
	if rowM(1,58) == 0
		W1 = [W1;rowM(1,1:57)];
	else
		W2 = [W2;rowM(1,1:57)];
	endif
endfor

rowsW1 = rows(W1);
rowsW2 = rows(W2);


# Calculating Prior-ratio
# =======================

P = (rowsW1/rows(M))/(rowsW2/rows(M));
####################

# Calculating values of mu, sigma for W1 and W2
# =============================================

mean1 = mean(W1);
vardiag1 = var(W1);
var1 = zeros(57,57);
for featno = [1:57]
	var1(featno, featno) = vardiag1(featno);
endfor

mean2 = mean(W2);
vardiag2 = var(W2);
var2 = zeros(57,57);
for featno = [1:57]
	var2(featno, featno) = vardiag2(featno);
endfor

# write the model into a file
# ===========================

csvwrite('full-bayes-parameters',[var1;mean1;var2;mean2]);
csvwrite('full-bayes-priors', P);
