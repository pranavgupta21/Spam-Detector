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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    GLOBAL VARIABLES     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# fetch the data set into a matrix
global D = csvread('spambase.data.txt');

global no_features = columns(D);
global n = 4;								# 4-fold validation as per the requirement of the assignment
global foldSize = floor(rows(D)/n);			# size of one fold


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        FUNCTIONS        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################## FUNCTION TO BUILD THE FULL BAYES MODEL #######################
################## ====================================== #######################


function LP = buildModel(M)
	global D n foldSize no_features;

	# divide the data of different classes
	# ====================================
	W1 = zeros(0,(no_features - 1));
	W2 = zeros(0,(no_features - 1));

	for rowno = [1:rows(M)]
		rowM = M(rowno,:);
		if rowM(1,no_features) == 0
			W1 = [W1;rowM(1,1:(no_features - 1))];
		else
			W2 = [W2;rowM(1,1:(no_features - 1))];
		endif
	endfor

	rowsW1 = rows(W1);
	rowsW2 = rows(W2);


	# Calculating Prior-ratio
	# =======================

	P = zeros(1,(no_features - 1));								# creating P as an array so it can be attached to LP, the function output
	P(1) = (rowsW1/rows(M))/(rowsW2/rows(M));
	####################

	# Calculating values of mu, sigma for W1 and W2
	# =============================================

	mean1 = mean(W1);
	vardiag1 = var(W1);
	var1 = zeros((no_features - 1),(no_features - 1));
	for featno = [1:(no_features - 1)]
		var1(featno, featno) = vardiag1(featno);
	endfor

	mean2 = mean(W2);
	vardiag2 = var(W2);
	var2 = zeros((no_features - 1),(no_features - 1));
	for featno = [1:(no_features - 1)]
		var2(featno, featno) = vardiag2(featno);
	endfor
	
	LP = [var1;mean1;var2;mean2;P];
endfunction

################# END OF FUNCTION BUILDMODEL() ###################
##################################################################



################## FUNCTION TO CLASSIFY THE TEST DATA #######################
################## ================================== #######################


function CM = detectSpam (M,LP)
	global D n foldSize no_features;
	CM = zeros(2,2);     			# confusion matrix - (1,1):tp, (2,2):tn, (1,2):fp, (2,1):fn
	RES = zeros(foldSize,1); 		# store 0 for non-spam and 1 for spam

	# Load the trained Model
	# ======================

	var1 = LP(1:(no_features - 1),:);
	mean1 = LP(no_features,:);

	var2 = LP((no_features + 1):(2 * no_features - 1),:);
	mean2 = LP(2 * no_features,:);

	P = LP((2 * no_features + 1),1);

	for rowno = [1:rows(M)]
		rowM = M(rowno,:);
	
		# calculate the decision surface function g(x) = g1(x) - g2(x); if g(x) > 0 : non-spam else spam
		rowMfeat = rowM(1:(no_features - 1));
		g1x = [(rowMfeat - mean2) * inverse(var1) * (rowMfeat - mean2)'];
		g2x = [(rowMfeat - mean2) * inverse(var2) * (rowMfeat - mean2)'];
	
		gx = g1x - g2x + P;
	
		# determine the class to be assigned
		if gx > 0
			C = 0;
			RES(rowno,1) = 0;
		else
			C = 1;
			RES(rowno,1) = 1;
		endif
		
		# determing tp,tn,fp,fn
		if C == 1 && rowM(1,no_features) == 1
			CM(1,1) = CM(1,1) + 1;
		elseif C == 0 && rowM(1,no_features) == 0
			CM(2,2) = CM(2,2) + 1;
		elseif C == 1 && rowM(1,no_features) == 0
			CM(1,2) = CM(1,2) + 1;
		elseif C == 0 && rowM(1,no_features) == 1
			CM(2,1) = CM(2,1) + 1;
		endif		
	endfor
endfunction

################# END OF FUNCTION DETECTSPAM() ###################
##################################################################



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        MAIN CODE        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# generate training and test ranges for n-cross validation
# ========================================================

CMG = zeros(2,2);						# confusion matrix

for testCaseNo = [1:n]
	trainData = zeros(0,no_features);
	testData = D((testCaseNo - 1) * foldSize + 1:testCaseNo*foldSize,:);
	for rangeNo = [1:n]
		if rangeNo != testCaseNo
			trainData = [trainData; D((rangeNo - 1) * foldSize + 1:rangeNo*foldSize,:)];
		endif
	endfor
	LP = buildModel(trainData);
	CMG = CMG .+ detectSpam(testData, LP);
endfor


# Calculate evaluation parameters
# ===============================

precision = CMG(1,1)/(CMG(1,1) + CMG(1,2));
recall = CMG(1,1)/(CMG(1,1) + CMG(2,1));
f1 = (2 * precision * recall)/(precision + recall);
macro = (CMG(1,1) + CMG(2,2))/(CMG(1,1) + CMG(2,2) + CMG(1,2) + CMG(2,1));
micro = (CMG(1,1) + CMG(1,2))/2;

# Display the results
# ===================
printf("\n\n");
printf("Confusion Matrix\n\n");
printf("\t| %5d | %5d |\n\t| %5d | %5d |\n\n",CMG(1,1), CMG(1,2), CMG(2,1), CMG(2,2));
printf("\nPRECISION  : %f",precision);
printf("\nRECALL     : %f",recall);
printf("\nF1-Measure : %f",f1);
printf("\nMACRO-AVG  : %f",macro);
printf("\nMICRO-AVG  : %f",micro);
printf("\n\n");


