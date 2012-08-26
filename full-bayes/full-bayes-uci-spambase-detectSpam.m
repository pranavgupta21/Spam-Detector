# Spam Detector based on full bayes decision boundary based classifier #
# uses the trained model stored in 'full-bayes-parameters' and developed using 'full-bayes-uci-spambase-buildModel.m'

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


# Load the trained Model
# ======================

M = csvread('full-bayes-parameters');
P = csvread('full-bayes-priors');

var1 = M(1:57,:);
mean1 = M(58,:);

var2 = M(59:115,:);
mean2 = M(116,:);

CM = zeros(2,2);     # confusion matrix - (1,1):tp, (2,2):tn, (1,2):fp, (2,1):fn
RES = zeros(1000,1);  # store 0 for non-spam and 1 for spam

# Run the classifier
# ==================

M = csvread('spambase.data.txt')(1:1150,:);

for rowno = [1:rows(M)]
	rowM = M(rowno,:);
	
	# calculate the decision surface function g(x) = g1(x) - g2(x); if g(x) > 0 : non-spam else spam
	rowMfeat = rowM(1:57);
	g1x = [(rowMfeat - mean2) * inverse(var1) * (rowMfeat - mean2)'];
	g2x = [(rowMfeat - mean2) * inverse(var2) * (rowMfeat - mean2)'];
	
	gx = g1x - g2x + P;
	
	# determine the class to be assigned
	if gx > 0
		C = 0;
	else
		C = 1;
	endif
		
	# determing tp,tn,fp,fn
	if C == 1 && rowM(1,58) == 1
		CM(1,1) = CM(1,1) + 1;
	elseif C == 0 && rowM(1,58) == 0
		CM(2,2) = CM(2,2) + 1;
	elseif C == 1 && rowM(1,58) == 0
		CM(1,2) = CM(1,2) + 1;
	elseif C == 0 && rowM(1,58) == 1
		CM(2,1) = CM(2,1) + 1;
	endif		
endfor

# Calculate evaluation parameters
# ===============================

accuracy = (CM(1,1) + CM(2,2))/(CM(1,1) + CM(2,2) + CM(1,2) + CM(2,1));
precision = CM(1,1)/(CM(1,1) + CM(1,2));
recall = CM(1,1)/(CM(1,1) + CM(2,1));
f1 = (2 * precision * recall)/(precision + recall);

# Display the results
# ===================
printf("\n\n");
printf("Confusion Matrix\n\n");
printf("\t| %5d | %5d |\n\t| %5d | %5d |\n\n",CM(1,1), CM(1,2), CM(2,1), CM(2,2));
printf("\nACCURACY : %f",accuracy);
printf("\nPRECISION : %f",precision);
printf("\nRECALL : %f",recall);
printf("\nF1-Measure : %f",f1);
printf("\n\n");

