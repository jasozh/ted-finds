from src.utils.utils import timer
from collections.abc import Callable
import numpy as np
import re


def tokenize(text: str):
    text = text.lower()
    return re.findall("[a-z]+", text)


def tokenize_transcript(tokenize_method, input_transcript):
    ret_list = []
    for utt in input_transcript[1]:
        text = utt['text']
        ret_list += tokenize_method(text)

    return ret_list


def num_dedup_tokens(tokenize_method, tokenize_transcript_method, input_transcripts):
    count = 0
    for transcript in input_transcripts:
        count += len(tokenize_transcript_method(tokenize_method, transcript))

    return count


def num_distinct_words(tokenize_method, tokenize_transcript_method, input_transcripts):
    accum = []
    for transcript in input_transcripts:
        accum += tokenize_transcript_method(tokenize_method, transcript)

    return len(set(accum))


def build_word_ted_talk_count(tokenize_method, tokenize_transcript_method, input_transcripts, input_titles):

    ted_talks = {}
    words = []

    for transcript in input_transcripts:
        uid = transcript[0]
        ted_talk_name = input_titles[uid]
        tokenized_transcript = tokenize_transcript_method(
            tokenize_method, transcript)
        words += tokenized_transcript
        if ted_talk_name in ted_talks:
            ted_talks[ted_talk_name] += tokenized_transcript
        else:
            ted_talks[ted_talk_name] = tokenized_transcript

    word_dict = {}
    words = set(words)

    for word in words:
        word_dict[word] = 0
        for ted_talk in ted_talks:
            if word in ted_talks[ted_talk]:
                word_dict[word] += 1

    return word_dict


def build_word_ted_talk_distribution(input_word_counts):
    numbers = list(set(input_word_counts.values()))
    dist = {}
    for number in numbers:
        dist[number] = list(input_word_counts.values()).count(number)

    return dist


def output_good_types(input_word_counts):
    ret_list = []
    for word in input_word_counts:
        if (input_word_counts[word] > 1):
            ret_list.append(word)
    return sorted(ret_list)


def create_ranked_good_types(tokenize_method, tokenize_transcript_method, input_transcripts, input_good_types):
    words_combined = []
    for transcript in input_transcripts:
        words = tokenize_transcript_method(tokenize_method, transcript)
        words_combined += words

    tup_list = []

    good_words = np.array(input_good_types)
    total_len = len(words_combined)

    for word in good_words:
        word_freq = round(words_combined.count(word) / total_len, 5)
        tup_list.append((word, word_freq))
    sorted_tups = sorted(tup_list, key=lambda x: x[1], reverse=True)
    return sorted_tups


def create_word_occurrence_matrix(tokenize_method, input_transcripts, input_speakers, input_good_types):
    speaker_indices = {speaker: idx for idx,
                       speaker in enumerate(input_speakers)}
    freqs = np.zeros((len(input_speakers), len(input_good_types)))

    for transcript in input_transcripts:
        for utt in transcript[1]:
            speaker = utt['speaker']
            if speaker in input_speakers:
                text = utt['text']
                tokens = tokenize_method(text)
                speaker_index = speaker_indices[speaker]
                arr = np.zeros(len(input_good_types))
                for word_index in range(len(input_good_types)):
                    good_word = input_good_types[word_index]
                    arr[word_index] = tokens.count(good_word)
                freqs[speaker_index] += arr[:]

    return freqs


def create_word_character_count_array(word_freq_matrix):
    return np.sum(word_freq_matrix[:] >= 1, axis=0)


def create_weighted_word_freq_array(input_word_array):
    sums = np.sum(input_word_array, axis=0) + 1
    oftens = input_word_array / sums
    return oftens
