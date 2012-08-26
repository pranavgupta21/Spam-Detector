# naive Bayes classifier on UCI spambase data set #
# assumption : the data is correct and preprocessed accurately without any anomalies #

# fetch the training data into a matrix
M = csvread('spambase.data.txt')(1151:4600,:);

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

P = [rowsW1/rows(M), rowsW2/rows(M)];
####################


# Calculating Likelihoods #
# ======================= #

w1fc = ones(57,10);       # initializing the count matrix with ones for smooting purpose
w1f = w1fc;
w2fc = ones(57,10);
w2f = w2fc;

f154ranges = [0 10 20 30 40 50 60 70 80 90 100];
f55ranges = [1 3 6];
f56ranges = [0 4 7];
f57ranges = [0 4 7];


# count matrix for w1
#=====================

for rowno = [1:rows(W1)]
	rowW1 = W1(rowno,:);
	rangeno = 0;
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
###################################

# count matrix for w2
#=====================

for rowno = [1:rows(W2)]
	rowW2 = W2(rowno,:);
	rangeno = 0;
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

#########################


# Likelihood Matrix for W1
# ========================

for rowno = [1:rows(w1fc)]
	for colno = [1:10]
		w1f(rowno, colno) = w1fc(rowno, colno)/(rowsW1+10);       # adding 10 since the count matrix was initialized with ones for smoothing
	endfor
endfor
##########################

# Likelihood Matrix for W2
# ========================

for rowno = [1:rows(w2fc)]
	for colno = [1:10]
		w2f(rowno, colno) = w2fc(rowno, colno)/(rowsW2+10);       # adding 10 since the count matrix was initialized with ones for smoothing
	endfor
endfor
##########################

########################################

# Write the Classifier to a file
# ==============================

csvwrite('naive-uci-spam-likelihoods', [w1f;w2f]);
csvwrite('naive-uci-spam-priors', P);
