from collections import defaultdict
import re

import numpy as np

"""
A python script to transform documents by replacing all mentions of co-referent clusters with first non-pronominal
mention.  Importantly, it explicitly handles nested coreferent mentions, which is very common and not handled by most
libraries.  This script, as of yet, does NOT handle syntax conflicts when replacing text.  I.e. if a possessive noun
is the head mention, it will be resolved indiscriminately regardless of contexts in which it is placed.  Please see
AllenNLP's function for guidance: https://docs.allennlp.org/models/master/models/coref/predictors/coref/#replace_corefs.
"""


PRONOUNS = {
    'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught', 'both', 'each other', 'each', 'either',
    'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his',
    'i', 'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my', 'myself', 'naught', 'neither', 'no one',
    'nobody', 'none', 'nothing', 'nought', 'one another', 'one', 'other', 'others', 'ought', 'our', 'ours', 'ourself',
    'ourselves', 'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat', 'such', 'suchlike', 'that',
    'thee', 'their', 'theirs', 'theirself', 'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they',
    'thine', 'this', 'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what', 'whatever', 'whatnot', 'whatsoever',
    'whence', 'where', 'whereby', 'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wheresoever', 'whereto',
    'whereunto', 'wherever', 'wherewith', 'wherewithal', 'whether', 'which', 'whichever', 'whichsoever', 'who',
    'whoever', 'whom', 'whomever', 'whomso', 'whomsoever', 'whose', 'whosesoever', 'whosever', 'whoso', 'whosoever',
    'ye', 'yon', 'yonder', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


def build_doc(resolved, spans, document):
    """
    :param resolved: ... dictionary of coreferent entities
    :param spans:
    :param document:
    :return:
    """
    curr_idx = 0
    toks = []
    while curr_idx < len(document):
        copy_to = -1
        for span in spans:
            if span[0] == curr_idx:
                copy_to = span[1]
                break

        if copy_to > -1:
            copy_str = span2str([curr_idx, copy_to])
            toks += resolved[copy_str]
            curr_idx = copy_to + 1
        else:
            toks.append(document[curr_idx])
            curr_idx += 1
    return toks


def build_str(main_span_str, resolved, dependencies, document, is_target):
    """
    :param span_str: string representation of span: {start}_{end}
    :param resolved: dictionary where keys are span_str's that have already been resolved to their final string
    representation
    :param dependencies: set of span_str's on which the resolution of main_span_str depends.
    There are 2 types of dependencies:
    1. replacement - if is_tgt, the span in dependencies (of which there should be just 1) is the head mention of
    the coreferent cluster which includes both
    2. subsumed - an entity (possibly part of a different cluster) is a subset of the main_span_str
    :param document:
    :param is_target: a boolean that is True if main_span_str is a non-head coreferent entity
    (i.e. it is a target of replacement).  This indicates its dependency is replacement, not subsumed
    :return: list of tokens.

    Resolves tokens from main_span_str in document according to the resolved tokens of spans on which it depends.
    """
    span = str2span(main_span_str)
    if is_target:
        copy_span_str = list(dependencies)[0]  # should only be one dependent (the head entity of the cluster)
        if copy_span_str in resolved:
            return resolved[copy_span_str]
        else:
            # They co-depend (for a few various reasons).
            # This is an artifact of coreference toolkits.
            # Just use target replacement from original document
            print('Warning. Circular dependency detected!')
            copy_span = str2span(copy_span_str)
            return document[copy_span[0]:copy_span[1] + 1]
    s = span[0]
    e = span[1]
    # remove sub-dependencies (sub-spans that are subsumed by larger spans which are also dependents.
    # The transformations are already resolved in spanning span)
    dep_spans = list(map(str2span, dependencies))
    dep_span_lens = list(map(span2len, dep_spans))
    dep_order = np.argsort(dep_span_lens)

    toks = []
    curr_idx = s
    while curr_idx <= e:
        copy_to = -1
        for dep_idx in dep_order:
            if dep_spans[dep_idx][0] == curr_idx:
                copy_to = dep_spans[dep_idx][1]
        if copy_to > -1:
            copy_str = span2str([curr_idx, copy_to])
            if copy_str in resolved:
                toks += resolved[copy_str]
            else:
                print('Warning. Circular dependency likely. Skipping.')
            curr_idx = copy_to + 1
        else:
            toks.append(document[curr_idx])
            curr_idx += 1

    return toks


def is_subset(span_small, span_big):
    """
    :param span_small: span array representing [start, end]
    :param span_big: span array representing [start, end]
    :return: boolean indicating iff a is fully contained within b
    """
    return span_small[0] >= span_big[0] and span_small[1] <= span_big[1]


def span2len(span):
    """
    :param span: span array representing [start, end]
    :return: span length where end index is inclusive (difference +1)
    """
    return span[1] - span[0] + 1


def span2str(span):
    """
    :param span: span array representing [start, end]
    :return: string version of span array
    """
    return '{}_{}'.format(str(span[0]), str(span[1]))


def span2toks(span, document):
    """
    :param span: span array representing [start, end]
    :param document: list of tokens from which to extract span tokens
    :return: tokens from document indicated by span indices
    """
    return document[span[0]:span[1] + 1]


def str2span(str):
    """
    :param str: string representation of span: {start}_{end}
    :return: span array representing [start, end]
    """
    s, e = str.split('_')
    return [int(s), int(e)]


def resolve(document, clusters):
    """
    :param document: list of tokens
    :param clusters: list of clusters where each cluster item is a text span whose indices refer to the document
    e.g. [
            [[0, 1], [2, 3]],
            [[4, 5], [8, 9]]
        ]
        Means that the text spans at [0, 1] and [2, 3] in document are part of a coreferent cluster.
        Likewise with [4, 5] and [8, 9]
    :return: Transformed document where all coreferent entities within a cluster are replaced with first pronominal
    reference.  Handles nested coreferent mentions by keeping track of entity dependencies and resolving entities
    in order.  I.e. resolve entities with no dependencies first before moving on to others.
    Circular coreferences are flagged and skipped (very rare - <1% in tests).
    """
    all_spans = set()

    span_dependencies = defaultdict(set)
    span_dependencies_rev = defaultdict(set)
    dep_counts = defaultdict(int)
    replaced_span_strs = set()
    subsumed_set = set()

    clusters_non_overlapping = []
    for cluster in clusters:
        cluster_starts = [c[0] for c in cluster]
        cluster_order = np.argsort(np.array(cluster_starts))

        cluster_non_overlapping = []
        for i, cidx in enumerate(cluster_order):
            curr_cluster = cluster[cidx]
            for j in range(i + 1, len(cluster)):
                if curr_cluster[1] >= cluster[cluster_order[j]][0]:
                    e_idx = max(curr_cluster[0], cluster[cluster_order[j]][0] - 1)
                    curr_cluster = [curr_cluster[0], e_idx]
                    break
            cluster_non_overlapping.append(curr_cluster)
        clusters_non_overlapping.append(cluster_non_overlapping)

        cluster_toks = list(map(lambda x: document[x[0]:x[1] + 1], cluster_non_overlapping))
        spans_no_pronouns = map(lambda x: set([y.lower() for y in x]) - PRONOUNS, cluster_toks)
        span_lens = np.array(list(map(len, spans_no_pronouns)))
        head_span_idx = None
        for i, span_len in enumerate(span_lens):
            if span_len > 0:
                head_span_idx = i
                break

        # No non-pronominal entities in cluster.  In this rare case, just select first reference in cluster.
        if head_span_idx is None:
            head_span_idx = 0

        head_span = cluster_non_overlapping[head_span_idx]
        head_span_str = span2str(head_span)
        head_name = ' '.join(cluster_toks[head_span_idx])
        for i, span in enumerate(cluster_non_overlapping):
            if i == head_span_idx:
                continue
            tgt_span_str = span2str(span)
            if tgt_span_str in replaced_span_strs:
                continue

            tgt_span_toks = cluster_toks[i]
            is_contained = re.search(re.escape(head_name.lower()), ' '.join(tgt_span_toks).lower()) is not None
            if is_contained:
                continue
            all_spans.add(head_span_str)
            all_spans.add(tgt_span_str)

            span_dependencies[tgt_span_str].add(head_span_str)
            span_dependencies_rev[head_span_str].add(tgt_span_str)
            dep_counts[tgt_span_str] = 1
            replaced_span_strs.add(tgt_span_str)

    all_spans = list(all_spans)
    all_span_lens = list(map(lambda x: span2len(str2span(x)), all_spans))
    order = np.argsort(np.array(all_span_lens))

    for i, span_idx in enumerate(order):
        small_span_str = all_spans[span_idx]
        small_span = str2span(small_span_str)
        for j in range(i, len(order)):
            other_span_idx = order[j]
            if other_span_idx == span_idx:
                continue
            big_span_str = all_spans[other_span_idx]
            big_span = str2span(big_span_str)
            if big_span_str in replaced_span_strs:
                continue
            is_sub = is_subset(small_span, big_span)
            if is_sub:
                subsumed_set.add(small_span_str)
            if is_sub and not big_span_str in replaced_span_strs:
                span_dependencies[big_span_str].add(small_span_str)
                span_dependencies_rev[small_span_str].add(big_span_str)
                dep_counts[big_span_str] += 1

    resolved = {}
    for _ in range(len(all_spans)):
        min_dep = 100000  # arbitrary large number
        min_span = None
        for span in all_spans:
            dc = dep_counts[span]
            if dc <= min_dep and span not in resolved:
                min_dep = dc
                min_span = span
        is_tgt = min_span in replaced_span_strs
        resolved[min_span] = build_str(min_span, resolved, span_dependencies[min_span], document, is_tgt)
        for child_span in span_dependencies_rev[min_span]:
            dep_counts[child_span] -= 1

    to_replace_span_strs = list(map(str2span, set(all_spans) - subsumed_set))
    resolved_toks = build_doc(resolved, to_replace_span_strs, document)
    return resolved_toks


if __name__ == '__main__':
    """
    This is JSON output from AllenNLP's Co-reference Resolution predictor.
    https://demo.allennlp.org/coreference-resolution
    
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.coref
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
    )
    document = "Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye
    Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two 
    years younger, with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their 
    high school, Lakeside, to develop their programming skills on several time-sharing computer systems."
    example = predictor.predict(document=document)
    
    """
    example = {
        'document': [
            'Paul', 'Allen', 'was', 'born', 'on', 'January', '21', ',', '1953', ',', 'in', 'Seattle', ',', 'Washington',
            ',', 'to', 'Kenneth', 'Sam', 'Allen', 'and', 'Edna', 'Faye', 'Allen', '.', 'Allen', 'attended', 'Lakeside',
            'School', ',', 'a', 'private', 'school', 'in', 'Seattle', ',', 'where', 'he', 'befriended', 'Bill',
            'Gates', ',', 'two', 'years', 'younger', ',', 'with', 'whom', 'he', 'shared', 'an', 'enthusiasm', 'for',
            'computers', '.', 'Paul', 'and', 'Bill', 'used', 'a', 'teletype', 'terminal', 'at', 'their', 'high',
            'school', ',', 'Lakeside', ',', 'to', 'develop', 'their', 'programming', 'skills', 'on', 'several',
            'time', '-', 'sharing', 'computer', 'systems', '.'
        ],
        'clusters': [
            [[0, 1], [24, 24], [36, 36], [47, 47], [54, 54]],
            [[11, 14], [33, 33]],
            [[38, 52], [56, 56]],
            [[54, 56], [62, 62], [70, 70]],
            [[26, 34], [62, 67]]
        ]
    }

    resolved_toks = resolve(example['document'], example['clusters'])
    print(' '.join(resolved_toks))

    """
    Prints the following (please note nested resolution):
    
    Paul Allen was born on January 21 , 1953 , in Seattle , Washington , to Kenneth Sam Allen and Edna Faye Allen .
    Paul Allen attended Lakeside School , a private school in Seattle , Washington , , where Paul Allen befriended
    Bill Gates , two years younger , with whom Paul Allen shared an enthusiasm for computers . Paul Allen and Bill Gates
    , two years younger , with whom Paul Allen shared an enthusiasm for computers used a teletype terminal at 
    Lakeside School , a private school in Seattle , Washington , , to develop Paul Allen and Bill Gates ,
    two years younger , with whom Paul Allen shared an enthusiasm for computers programming skills on several time -
    sharing computer systems .
    
    """
