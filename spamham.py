import math
import argparse
import csv

spam_array = []
ham_array = []
spamword_dict = {}
hamword_dict = {}
spam_number = 0
ham_number = 0
num_of_freq_in_ham = 0
num_of_freq_in_spam = 0
vocabulary = []
laplace_smoothing_factor = 1
# A list of common stop-words
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
              "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
              "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
              "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves"]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-f1', help='train_dataset')
parser.add_argument('-f2', help='test_dataset')
parser.add_argument('-o', help='output_file')
args = parser.parse_args()
f = open(args.f1, "r")
# Parse train data and split spam and ham information
for line in f:
    data = line.split()
    if data[1] == 'spam':
        spam_number = spam_number + 1
        spam_array.append(line)

    elif data[1] == 'ham':
        ham_number = ham_number + 1
        ham_array.append(line)
f.close()

# Calculating prior probabilities for spam and ham
pc_for_spam = float(spam_number) / float((spam_number + ham_number))
pc_for_ham = float(ham_number) / float((spam_number + ham_number))
print "Prior Probability for spam = ", pc_for_spam
print "Prior Probability for ham = ", pc_for_ham

vocabulary_dict = {}
# Parsing through train data and constructing dictionary for spam and construct vocabulary
print "Parsing through spam training data..."
for line in spam_array:
    token_split = line.split()
    del token_split[0:2]
    j = 0
    # Go through each token and add the frequency to the dictionary
    while j < len(token_split):
        if token_split[j] in spamword_dict.keys():
            spamword_dict[token_split[j]] = spamword_dict[token_split[j]] + int(token_split[j + 1])
        else:
            spamword_dict[token_split[j]] = int(token_split[j + 1])
        num_of_freq_in_spam = num_of_freq_in_spam + int(token_split[j + 1])
        # Add each value to vocabulary if not present
        if token_split[j] not in vocabulary:
            vocabulary.append(token_split[j])
            vocabulary_dict[token_split[j]] = 1
        else:
            vocabulary_dict[token_split[j]] = vocabulary_dict[token_split[j]] + 1
        j = j + 2

num_of_term_in_spam = len(spamword_dict)

# Parsing through train data and constructing dictionary for ham
print "Parsing through ham training data..."
for line in ham_array:
    token_split = line.split()
    del token_split[0:2]
    # print token_split
    j = 0
    # Go through each token and add the frequency to the dictionary
    while j < len(token_split):
        if token_split[j] in hamword_dict.keys():
            hamword_dict[token_split[j]] = hamword_dict[token_split[j]] + int(token_split[j + 1])
        else:
            hamword_dict[token_split[j]] = int(token_split[j + 1])
        num_of_freq_in_ham = num_of_freq_in_ham + int(token_split[j + 1])
        # Add each value to vocabulary if not present
        if token_split[j] not in vocabulary:
            vocabulary.append(token_split[j])
            vocabulary_dict[token_split[j]] = 1
        else:
            vocabulary_dict[token_split[j]] = vocabulary_dict[token_split[j]] + 1
        j = j + 2

num_of_term_in_ham = len(hamword_dict)
num_of_term_in_vocabulary = len(vocabulary)

# Parse through test data and calculate term conditional probabilities using stopwords, idf and smoothing parameter
print "Parsing through test data..."
incorrect_ans = 0
correct_ans = 0
f = open(args.f2, "r")
decision = []
for line in f:
    token_split = line.split()
    email_decision = []
    email_decision.insert(0, token_split[0])
    expected_val = token_split[1]
    del token_split[0:2]
    j = 0
    probability_ham = 1.0
    # Go through each token and calculate term probabilities based on our training data
    # First check the probability of ham
    while j < len(token_split):
        if token_split[j] in stop_words:
            j = j + 2
            continue
        idf = math.log((spam_number + ham_number) / vocabulary_dict[token_split[j]])
        if token_split[j] in hamword_dict.keys():
            probability_ham = probability_ham + math.log(float(
                float((hamword_dict[token_split[j]] * idf) + laplace_smoothing_factor) /
                (float(num_of_freq_in_ham + (laplace_smoothing_factor * num_of_term_in_vocabulary)))), 10)
        else:
            probability_ham = probability_ham + math.log(float(
                float(laplace_smoothing_factor) /
                (float(num_of_freq_in_ham + (laplace_smoothing_factor * num_of_term_in_vocabulary)))), 10)
        # print probability
        j = j + 2

    j = 0
    probability_spam = 1.0
    # Checking the probability of spam
    while j < len(token_split):
        if token_split[j] in stop_words:
            j = j + 2
            continue
        idf = math.log((spam_number + ham_number) / vocabulary_dict[token_split[j]])
        if token_split[j] in spamword_dict.keys():
            probability_spam = probability_spam + math.log(float(
                float((spamword_dict[token_split[j]] * idf) + laplace_smoothing_factor) /
                (float(num_of_freq_in_ham + (laplace_smoothing_factor * num_of_term_in_vocabulary)))), 10)
        else:
            probability_spam = probability_spam + math.log(float(
                float(laplace_smoothing_factor) /
                (float(num_of_freq_in_ham + (laplace_smoothing_factor * num_of_term_in_vocabulary)))), 10)
        j = j + 2

    if (probability_spam + math.log(pc_for_spam, 10)) > (probability_ham + math.log(pc_for_ham, 10)):
        calc_valu = "spam"
    else:
        calc_valu = "ham"
    if calc_valu == expected_val:
        correct_ans = correct_ans + 1
    else:
        incorrect_ans = incorrect_ans + 1
    email_decision.insert(1, calc_valu)
    decision.append(email_decision)

# Writing output to CSV file
output = open(args.o, 'w')
with output:
    write_object = csv.writer(output)
    write_object.writerows(decision)

# Calculate accuracy for the predictions
accuracy = (float(correct_ans) / (correct_ans + incorrect_ans)) * 100
print "Correct analysis", correct_ans
print "Incorrect analysis", incorrect_ans
print "Decision: ", decision
print "Accuracy", accuracy
