# naive Bayes classifier on UCI spambase data set #
# assumption : the data is correct and preprocessed accurately without any anomalies #

# load test data
M = csvread('spambase.data.txt')(1:1150,:);

# load Model

W1 = csvread('naive-uci-spam-likelihoods')(1:57,:);
W2 = csvread('naive-uci-spam-likelihoods')(58:114,:);

P = csvread('naive-uci-spam-priors');

CM = zeros(2,2);     # confusion matrix - (1,1):tp, (2,2):tn, (1,2):fp, (2,1):fn
RES = zeros(1000,1);  # store 0 for non-spam and 1 for spam

f154ranges = [0 10 20 30 40 50 60 70 80 90 100];
f55ranges = [1 3 6];
f56ranges = [0 4 7];
f57ranges = [0 4 7];

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

