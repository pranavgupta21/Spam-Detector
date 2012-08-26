# naive Bayes classifier on UCI spambase data set #
# assumption : the data is correct and preprocessed accurately without any anomalies #

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    GLOBAL VARIABLES     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# fetch the data set into a matrix
global D = csvread('spambase.data.txt');

global n = 4;								# 4-fold validation as per the requirement of the assignment
global foldSize = floor(rows(D)/n);			# size of one fold

# Discretization of the features
global f154ranges = [0 10 20 30 40 50 60 70 80 90 100];
global f55ranges = [1 3 6];
global f56ranges = [0 4 7];
global f57ranges = [0 4 7];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        FUNCTIONS        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################## FUNCTION TO BUILD THE NAIVE BAYES MODEL #######################
################## ======================================= #######################


function LP = buildModel(M)
	global D n foldSize f154ranges f55ranges f56ranges f57ranges;

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


	# Calculating Priors
	# ==================

	P = zeros(1,10);
	P(1) = rowsW1/rows(M);
	P(2) = rowsW2/rows(M);
	####################


	# Calculating Likelihoods #
	# ======================= #

	w1fc = ones(57,10);       # initializing the count matrix with ones for smooting
	w1f = w1fc;
	w2fc = ones(57,10);
	w2f = w2fc;

	# count matrix for w1
	#=====================

	for rowno = [1:rows(W1)]
		rowW1 = W1(rowno,:);
		# colums(features) 1 - 54 : continuous [1,100]
		for colno = [1:54]
			# make cases for discretization
			for rangeno = [1:columns(f154ranges)]
				if rowW1(1,colno) >=f154ranges(rangeno) && rowW1(1,colno) < f154ranges(rangeno+1)
					w1fc(colno,rangeno) = w1fc(colno,rangeno) + 1;
					break;
				endif
			endfor
		endfor
	
		# feature 55 : continuous real [1 .. ]
		for rangeno = [1:(columns(f55ranges)-1)]
			if rowW1(1,55) >= f55ranges(rangeno) && rowW1(1,55) < f55ranges(rangeno+1)
				w1fc(55,rangeno) = w1fc(55,rangeno) + 1;
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowW1(1,55) >= f55ranges(rangeno)
			w1fc(55,rangeno) = w1fc(55,rangeno) + 1;
		endif	

		# feature 56 : continuous integer [1 .. ]
		for rangeno = [1:(columns(f56ranges)-1)]
			if rowW1(1,56) >= (f56ranges(rangeno)+1) && rowW1(1,56) < f56ranges(rangeno+1)
				w1fc(56,rangeno) = w1fc(56,rangeno) + 1;
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowW1(1,56) >= (f56ranges(rangeno)+1)
			w1fc(56,rangeno) = w1fc(56,rangeno) + 1;
		endif	

		# feature 57 : continuous integer [1 .. ]
		for rangeno = [1:(columns(f57ranges)-1)]
			if rowW1(1,57) >= (f57ranges(rangeno)+1) && rowW1(1,57) < f57ranges(rangeno+1)
				w1fc(57,rangeno) = w1fc(57,rangeno) + 1;
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowW1(1,57) >= (f57ranges(rangeno)+1)
			w1fc(57,rangeno) = w1fc(57,rangeno) + 1;
		endif	
	endfor

	# count matrix for w2
	# ===================

	for rowno = [1:rows(W2)]
		rowW2 = W2(rowno,:);
		# colums(features) 1 - 54 : continuous [1,100]
		for colno = [1:54]
			# make cases for discretization
			for rangeno = [1:columns(f154ranges)]
				if rowW2(1,colno) >=f154ranges(rangeno) && rowW2(1,colno) < f154ranges(rangeno+1)
					w2fc(colno,rangeno) = w2fc(colno,rangeno) + 1;
					break;
				endif
			endfor
		endfor
	
		# feature 55 : continuous real [1 .. ]
		for rangeno = [1:(columns(f55ranges)-1)]
			if rowW2(1,55) >= f55ranges(rangeno) && rowW2(1,55) < f55ranges(rangeno+1)
				w2fc(55,rangeno) = w2fc(55,rangeno) + 1;
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowW2(1,55) >= f55ranges(rangeno)
			w2fc(55,rangeno) = w2fc(55,rangeno) + 1;
		endif	
			
		# feature 56 : continuous integer [1 .. ]
		for rangeno = [1:(columns(f56ranges)-1)]
			if rowW2(1,56) >= (f56ranges(rangeno)+1) && rowW2(1,56) < f56ranges(rangeno+1)
				w2fc(56,rangeno) = w2fc(56,rangeno) + 1;
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowW2(1,56) >= (f56ranges(rangeno)+1)
			w2fc(56,rangeno) = w2fc(56,rangeno) + 1;
		endif	

		# feature 57 : continuous integer [1 .. ]
		for rangeno = [1:(columns(f57ranges)-1)]
			if rowW2(1,57) >= (f57ranges(rangeno)+1) && rowW2(1,57) < f57ranges(rangeno+1)
				w2fc(57,rangeno) = w2fc(57,rangeno) + 1;
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowW2(1,57) >= (f57ranges(rangeno)+1)
			w2fc(57,rangeno) = w2fc(57,rangeno) + 1;	
		endif	
	endfor	


	# Likelihood Matrix for W1
	# ========================

	for rowno = [1:rows(w1fc)]
		for colno = [1:10]
			w1f(rowno, colno) = w1fc(rowno, colno)/(rowsW1+10);       # adding 10 to normalize smoothing
		endfor
	endfor

	# Likelihood Matrix for W2
	# ========================

	for rowno = [1:rows(w2fc)]
		for colno = [1:10]
			w2f(rowno, colno) = w2fc(rowno, colno)/(rowsW2+10);       # adding 10 to normalize laplace (add-1) smoothing
		endfor
	endfor

	LP = [w1f;w2f;P];
endfunction

################# END OF FUNCTION BUILDMODEL() ###################
##################################################################



################## FUNCTION TO CLASSIFY THE TEST DATA #######################
################## ================================== #######################


function CM = detectSpam (M,L,P)
	global D n foldSize f154ranges f55ranges f56ranges f57ranges;
	CM = zeros(2,2);     # confusion matrix - (1,1):tp, (2,2):tn, (1,2):fp, (2,1):fn
	RES = zeros(foldSize,1);  # store 0 for non-spam and 1 for spam

	#csvwrite('naive-uci-spam-likelihoods', L);
	
	W1 = L(1:57,:);
	W2 = L(58:114,:);
	
	for rowno = [1:rows(M)]
		rowM = M(rowno,:);
		G = [1 1];
		# multiply likelihoods of all features (assuming that the features are independent)
		# features 1 - 54 : continuous [1,100]
		rangeno = 0;
		for colno = [1:54]
			# make cases for discretization
			for rangeno = [1:columns(f154ranges)]
				if rowM(1,colno) >=f154ranges(rangeno) && rowM(1,colno) < f154ranges(rangeno+1)
					G(1) = G(1) * W1(colno,rangeno);
					G(2) = G(2) * W2(colno,rangeno);
					break;
				endif
			endfor
		endfor
	
		# feature 55 : continuous real [1 .. ]
		for rangeno = [1:(columns(f55ranges)-1)]
			if rowM(1,55) >= f55ranges(rangeno) && rowM(1,55) < f55ranges(rangeno+1)
				G(1) = G(1) * W1(colno,rangeno);
				G(2) = G(2) * W2(colno,rangeno);
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowM(1,55) >= f55ranges(rangeno)
				G(1) = G(1) * W1(colno,rangeno);
				G(2) = G(2) * W2(colno,rangeno);
		endif	

		# feature 56 : continuous integer [1 .. ]
		for rangeno = [1:(columns(f56ranges)-1)]
			if rowM(1,56) >= (f56ranges(rangeno)+1) && rowM(1,56) < f56ranges(rangeno+1)
				G(1) = G(1) * W1(colno,rangeno);
				G(2) = G(2) * W2(colno,rangeno);
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowM(1,56) >= (f56ranges(rangeno)+1)
				G(1) = G(1) * W1(colno,rangeno);
				G(2) = G(2) * W2(colno,rangeno);
		endif	

		# feature 57 : continuous integer [1 .. ]
		for rangeno = [1:(columns(f57ranges)-1)]
			if rowM(1,57) >= (f57ranges(rangeno)+1) && rowM(1,57) < f57ranges(rangeno+1)
				G(1) = G(1) * W1(colno,rangeno);
				G(2) = G(2) * W2(colno,rangeno);
				break;
			endif
		endfor
		rangeno = rangeno + 1;
		if rowM(1,57) >= (f57ranges(rangeno)+1)
				G(1) = G(1) * W1(colno,rangeno);
				G(2) = G(2) * W2(colno,rangeno);
		endif	
	
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
	trainData = zeros(0,58);
	testData = D((testCaseNo - 1) * foldSize + 1:testCaseNo*foldSize,:);
	for rangeNo = [1:n]
		if rangeNo != testCaseNo
			trainData = [trainData; D((rangeNo - 1) * foldSize + 1:rangeNo*foldSize,:)];
		endif
	endfor
	LP = buildModel(trainData);
	L = LP(1:114,:);
	P = LP(115,1:2);
	CMG = CMG .+ detectSpam(testData, L, P);
endfor


# Calculate evaluation parameters
# ===============================

accuracy = (CMG(1,1) + CMG(2,2))/(CMG(1,1) + CMG(2,2) + CMG(1,2) + CMG(2,1));
precision = CMG(1,1)/(CMG(1,1) + CMG(1,2));
recall = CMG(1,1)/(CMG(1,1) + CMG(2,1));
f1 = (2 * precision * recall)/(precision + recall);

# Display the results
# ===================
printf("\n\n");
printf("Confusion Matrix\n\n");
printf("\t| %5d | %5d |\n\t| %5d | %5d |\n\n",CMG(1,1), CMG(1,2), CMG(2,1), CMG(2,2));
printf("\nACCURACY : %f",accuracy);
printf("\nPRECISION : %f",precision);
printf("\nRECALL : %f",recall);
printf("\nF1-Measure : %f",f1);
printf("\n\n");



