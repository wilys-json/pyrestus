# Similarity measures

def similarity(seq1, seq2, method='dice'):

    valid_methods = ['dice', 'jaccard', 'overlap_coeff']
    assert method in valid_methods, \
        f"only provide the following methods: {valid_methods}; got {method}"

    seq1 = set(seq1)
    seq2 = set(se2)

    intersection = len(seq1.intersection(seq2))
    if method =='dice':
        intersection *= 2

    denom = {
        "dice" : len(seq1) + len(seq2),
        "jaccard" : len(seq1 & seq2),
        "overlap_coeff" : min(len(seq1), len(seq2))
    }

    return intersection / denom['method']
