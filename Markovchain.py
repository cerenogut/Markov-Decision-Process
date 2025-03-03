import numpy as np
import matplotlib.pyplot as plt
import random

# GLOBALS
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")  
END_SYMBOL = '*'
ALL_STATES = ALPHABET + [END_SYMBOL]  

def index_of_letter(letter):
    return ALL_STATES.index(letter)


def dict_increment(d, key, inc=1):
    if key in d:
        d[key] += inc
    else:
        d[key] = inc

def dict2_increment(d, key1, key2, inc=1):
    if key1 not in d:
        d[key1] = {}
    if key2 not in d[key1]:
        d[key1][key2] = 0
    d[key1][key2] += inc

def read_words_from_file(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w.lower() + END_SYMBOL)
    return words

def estimate_probs_first_order(words):
    first_letter_counts = {}
    transition_counts = {}  
    total_after_letter = {} 

    total_words = len(words)

    for w in words:
        chars = list(w)
        if len(chars) == 0:
            continue

        dict_increment(first_letter_counts, chars[0], 1)

        for i in range(len(chars) - 1):
            curr_char = chars[i]
            next_char = chars[i+1]
            dict2_increment(transition_counts, curr_char, next_char, 1)
            dict_increment(total_after_letter, curr_char, 1)

    P_L0 = {}
    for letter in ALL_STATES:
        c = first_letter_counts[letter] if letter in first_letter_counts else 0
        P_L0[letter] = c / total_words if total_words > 0 else 0.0

    P_LN_given_LNminus1 = np.zeros((27, 27))
    for i, letter1 in enumerate(ALL_STATES):
        denom = total_after_letter[letter1] if letter1 in total_after_letter else 0
        for j, letter2 in enumerate(ALL_STATES):
            count_ij = 0
            if letter1 in transition_counts:
                if letter2 in transition_counts[letter1]:
                    count_ij = transition_counts[letter1][letter2]
            if denom > 0:
                P_LN_given_LNminus1[i, j] = count_ij / denom
            else:
                P_LN_given_LNminus1[i, j] = 0.0

    return P_L0, P_LN_given_LNminus1

def calculate_avg_word_length(words):
    if len(words) == 0:
        return 0.0
    total_len = 0
    for w in words:
        total_len += (len(w) - 1)  
    return total_len / len(words)

def calculatePriorProb1(words, N):
    count_LN = {}
    total_words = len(words)

    for w in words:
        idx = N - 1
        if idx < len(w):
            letterN = w[idx]  
            dict_increment(count_LN, letterN, 1)

    # normalize
    P_LN = {}
    for letter in ALL_STATES:
        c = count_LN[letter] if letter in count_LN else 0
        P_LN[letter] = c / total_words if total_words > 0 else 0.0

    return P_LN

def plot_distrib(distribution, title):
    labels = ALL_STATES
    values = [distribution[l] for l in labels]
    plt.figure(figsize=(10,4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Harfler (+ '*')")
    plt.ylabel("Olasılık")
    plt.show()

def calculatePriorProb2(P_L0, P_LN_given_LNminus1, N):
    P_current = np.array([P_L0[l] for l in ALL_STATES]) 
    for _ in range(N):
        P_next = P_current.dot(P_LN_given_LNminus1)  
        P_current = P_next

    P_LN = {}
    for i, letter in enumerate(ALL_STATES):
        P_LN[letter] = P_current[i]
    return P_LN

def calculateWordProb(P_L0, P_LN_given_LNminus1, word):
    chars = list(word)
    if len(chars) == 0:
        return 0.0

    if chars[0] not in P_L0:
        return 0.0
    prob = P_L0[chars[0]]

    for i in range(len(chars) - 1):
        l_prev = chars[i]
        l_next = chars[i+1]
        row_idx = index_of_letter(l_prev)
        col_idx = index_of_letter(l_next)
        p_trans = P_LN_given_LNminus1[row_idx, col_idx]
        prob *= p_trans
        if prob == 0.0:
            break

    return prob

def generate_words(P_L0, P_LN_given_LNminus1, M):
    letters_array = np.array(ALL_STATES)
    p_l0_array = np.array([P_L0[l] for l in ALL_STATES])

    generated_words = []
    for _ in range(M):
        # 1st letter:
        current_letter = np.random.choice(letters_array, p=p_l0_array)
        word_chars = [current_letter]

        # following letters:
        while current_letter != END_SYMBOL:
            row_idx = index_of_letter(current_letter)
            transition_probs = P_LN_given_LNminus1[row_idx, :]
            next_letter = np.random.choice(letters_array, p=transition_probs)
            word_chars.append(next_letter)
            current_letter = next_letter

        generated_words.append("".join(word_chars))

    return generated_words

def synthetic_avg_word_length(P_L0, P_LN_given_LNminus1, sample_size=100000):
    gen_words = generate_words(P_L0, P_LN_given_LNminus1, sample_size)
    total_len = 0
    for w in gen_words:
        total_len += (len(w) - 1)  
    return total_len / sample_size

#----------------------BONUS PART: 2nd ORDER MARKOV---------------------------------------

def estimate_probabilities_second_order(words):
    transitions = {}  
    total_counts = {} 

    for w in words:
        chars = list(w)
        for i in range(len(chars) - 2):
            pair = (chars[i], chars[i+1])
            next_char = chars[i+2]

            if pair not in transitions:
                transitions[pair] = {}
            if next_char not in transitions[pair]:
                transitions[pair][next_char] = 0
            transitions[pair][next_char] += 1

            if pair not in total_counts:
                total_counts[pair] = 0
            total_counts[pair] += 1

    P_2nd = {}
    for pair in transitions:
        pair_dict = {}
        denom = total_counts[pair]
        for letter in ALL_STATES:
            count_val = transitions[pair][letter] if letter in transitions[pair] else 0
            if denom > 0:
                pair_dict[letter] = count_val / denom
            else:
                pair_dict[letter] = 0.0
        P_2nd[pair] = pair_dict

    return P_2nd

def generate_words_second_order(words, M):
    pair_counts = {}
    total_pairs = 0
    for w in words:
        chars = list(w)
        if len(chars) >= 2:
            pair_0 = (chars[0], chars[1])
            dict_increment(pair_counts, pair_0, 1)
            total_pairs += 1

    pairs_list = list(pair_counts.keys())
    pair_probs = []
    for p in pairs_list:
        pair_probs.append(pair_counts[p] / total_pairs if total_pairs>0 else 0.0)

    P_2nd = estimate_probabilities_second_order(words)

    generated = []
    for _ in range(M):
        chosen_pair = random.choices(pairs_list, weights=pair_probs, k=1)[0]
        (c0, c1) = chosen_pair
        word_chars = [c0, c1]

        while True:
            pair_key = (word_chars[-2], word_chars[-1])
            if pair_key not in P_2nd:
                word_chars.append(END_SYMBOL)
                break

            dist_dict = P_2nd[pair_key]
            letters_array = np.array(ALL_STATES)
            probs_array = np.array([dist_dict[l] for l in ALL_STATES])

            next_char = np.random.choice(letters_array, p=probs_array)
            word_chars.append(next_char)

            if next_char == END_SYMBOL:
                break

        generated.append("".join(word_chars))

    return generated

#---------------BONUS PART: 3rd ORDER MARKOV--------------------------------------

def estimate_probabilities_third_order(words):
    transitions = {}   
    total_counts_3 = {}  
    for w in words:
        chars = list(w)
        for i in range(len(chars) - 3):
            triple = (chars[i], chars[i+1], chars[i+2])
            next_char = chars[i+3]

            if triple not in transitions:
                transitions[triple] = {}
            if next_char not in transitions[triple]:
                transitions[triple][next_char] = 0
            transitions[triple][next_char] += 1

            if triple not in total_counts_3:
                total_counts_3[triple] = 0
            total_counts_3[triple] += 1

    P_3rd = {}
    for triple in transitions:
        triple_dict = {}
        denom = total_counts_3[triple]
        for letter in ALL_STATES:
            if letter in transitions[triple]:
                count_val = transitions[triple][letter]
            else:
                count_val = 0
            if denom > 0:
                triple_dict[letter] = count_val / denom
            else:
                triple_dict[letter] = 0.0
        P_3rd[triple] = triple_dict

    return P_3rd

def generate_words_third_order(words, M):

    triple_counts = {}
    total_triples = 0
    for w in words:
        chars = list(w)
        if len(chars) >= 3:
            triple_0 = (chars[0], chars[1], chars[2])
            dict_increment(triple_counts, triple_0, 1)
            total_triples += 1

    triple_list = list(triple_counts.keys())
    triple_probs = []
    for t in triple_list:
        triple_probs.append(triple_counts[t] / total_triples if total_triples > 0 else 0.0)

    P_3rd = estimate_probabilities_third_order(words)

    generated = []
    for _ in range(M):        
        chosen_triple = random.choices(triple_list, weights=triple_probs, k=1)[0]
        (c0, c1, c2) = chosen_triple
        word_chars = [c0, c1, c2]

        while True:
            triple_key = (word_chars[-3], word_chars[-2], word_chars[-1])
            if triple_key not in P_3rd:
                word_chars.append(END_SYMBOL)
                break

            dist_dict = P_3rd[triple_key]
            letters_array = np.array(ALL_STATES)
            probs_array = np.array([dist_dict[l] for l in ALL_STATES])

            next_char = np.random.choice(letters_array, p=probs_array)
            word_chars.append(next_char)

            if next_char == END_SYMBOL:
                break

        generated.append("".join(word_chars))

    return generated

#-----------------MAIN-----------------------

if __name__ == "__main__":
    filename = "corncob_lowercase.txt" 
    words = read_words_from_file(filename)
    print(f"Total word count (data)): {len(words)}")

    # 1) P(L0), P(LN|LN-1):
    P_L0, P_LN_given_LNminus1 = estimate_probs_first_order(words)
    print("\n--- P(L0) ---")
    for k in ALL_STATES:
        print(f"{k}: {P_L0[k]:.6f}")

    print("\n--- P(LN|LN-1) matris size ---")
    print(P_LN_given_LNminus1.shape)

    # 2) Average word length (data):
    avg_len = calculate_avg_word_length(words)
    print(f"\nAverage word length in the dataset: {avg_len:.4f}")

    # 3) calculatePriorProb1 and plot:
    for N in [1, 2, 3, 4, 5]:
        distN = calculatePriorProb1(words, N)
        plot_distrib(distN, f"calculatePriorProb1 -> P(L{N})")

    # 4) calculatePriorProb2 and plot:
    for N in [1, 2, 3, 4, 5]:
        distN = calculatePriorProb2(P_L0, P_LN_given_LNminus1, N)
        plot_distrib(distN, f"calculatePriorProb2 -> P(L{N})")

    # 5) calculateWordProb:
    test_words = ["sad*", "exchange*", "antidisestablishmentarianism*", 
                  "qwerty*", "zzzz*", "ae*"]
    print("\nProbabilities of words (1st order):")
    for tw in test_words:
        p_tw = calculateWordProb(P_L0, P_LN_given_LNminus1, tw)
        print(f"{tw} -> {p_tw:.12f}")

    # 6) generate_words (10):
    gen_10 = generate_words(P_L0, P_LN_given_LNminus1, 10)
    print("\n10 generated words (1st order):")
    for gw in gen_10:
        print(gw)

    # 7) Average length of 100000 synthetic words:
    synthetic_avg_len = synthetic_avg_word_length(P_L0, P_LN_given_LNminus1, sample_size=100000)
    print(f"\nAverage length for 100000 synthetic words: {synthetic_avg_len:.4f}")

    #-----------------BONUS (2nd Order)----------
    gen_10_second = generate_words_second_order(words, 10)
    print("\n10 generated words (2nd order):")
    for gw in gen_10_second:
        print(gw) 

    #-----------------BONUS (3rd Order)----------
    gen_10_third = generate_words_third_order(words, 10)
    print("\n10 generated words (3rd order):")
    for gw in gen_10_third:
        print(gw) 
