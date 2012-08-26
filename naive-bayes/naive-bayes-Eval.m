# naive Bayes classifier on UCI spambase data set #
# assumption : the data is correct and preprocessed accurately without any anomalies #

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


################## FUNCTION TO BUILD THE NAIVE BAYES MODEL #######################
################## ======================================= #######################


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


	# Calculating Priors
	# ==================

	P = [rowsW1/rows(M), rowsW2/rows(M)];
	####################


	# Calculating Likelihoods #
	# ======================= #

	w1fc = ones((no_features - 1),2);       # initializing the count matrix with ones for smooting
	w1f = w1fc;
	w2fc = ones((no_features - 1),2);
	w2f = w2fc;

	# count matrix for w1
	#=====================

	for rowno = [1:rows(W1)]
		rowW1 = W1(rowno,:);
		for colno = [1:(no_features - 1)]
			if rowW1(1,colno) == 0
				w1fc(colno,1) = w1fc(colno,1) + 1;
			else
				w1fc(colno,2) = w1fc(colno,2) + 1;
			endif
		endfor
	endfor

	# count matrix for w2
	# ===================

	for rowno = [1:rows(W2)]
		rowW2 = W2(rowno,:);
		for colno = [1:(no_features - 1)]
			if rowW2(1,colno) == 0
				w2fc(colno,1) = w2fc(colno,1) + 1;
			else
				w2fc(colno,2) = w2fc(colno,2) + 1;
			endif
		endfor
	endfor

	# Likelihood Matrix for W1
	# ========================

	for rowno = [1:rows(w1fc)]
		w1f(rowno, 1) = w1fc(rowno, 1)/(rowsW1+2);       # adding 2 to normalize smoothing
		w1f(rowno, 2) = w1fc(rowno, 2)/(rowsW1+2);       # adding 2 to normalize smoothing
	endfor

	# Likelihood Matrix for W2
	# ========================

	for rowno = [1:rows(w2fc)]
		w2f(rowno, 1) = w2fc(rowno, 1)/(rowsW2+2);       # adding 2 to normalize smoothing
		w2f(rowno, 2) = w2fc(rowno, 2)/(rowsW2+2);       # adding 2 to normalize smoothing
	endfor

	LP = [w1f;w2f;P];
endfunction

################# END OF FUNCTION BUILDMODEL() ###################
##################################################################



################## FUNCTION TO CLASSIFY THE TEST DATA #######################
################## ================================== #######################


function CM = detectSpam (M,L,P)
	global D n foldSize no_features;
	CM = zeros(2,2);     # confusion matrix - (1,1):tp, (2,2):tn, (1,2):fp, (2,1):fn
	RES = zeros(foldSize,1);  # store 0 for non-spam and 1 for spam

	#csvwrite('naive-uci-spam-likelihoods', L);
	
	W1 = L(1:(no_features - 1),:);
	W2 = L(no_features:(2 * (no_features - 1)),:);
	
	for rowno = [1:rows(M)]
		rowM = M(rowno,:);
		G = [1 1];
		# multiply likelihoods of all features (assuming that the features are independent)
		for colno = [1:(no_features - 1)]
			if rowM(1,colno) == 0 
				G(1) = G(1) * W1(colno,1);
				G(2) = G(2) * W2(colno,1);
			else
				G(1) = G(1) * W1(colno,2);
				G(2) = G(2) * W2(colno,2);
			endif
		endfor
	
		# multiply the priors to calculate final probabilities
		G(1) = G(1) * P(1,1);
		G(2) = G(2) * P(1,2);
	
		# determine the class label to be assigned
		C = 0;
		if G(1) > G(2)
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
	L = LP(1:(2 * (no_features - 1)),:);
	P = LP((2 * (no_features - 1)) + 1,:);
	CMG = CMG .+ detectSpam(testData, L, P);
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


