from pathlib import Path
from phonet.phonet import Phonet  # type: ignore --> to supress pylance problem
from scipy import signal
import numpy as np
LEXICON = {
    'phones': [
        '<p:>', 'B', 'D', 'F', 'G', 'J', 'L', 'N', 'S', 'T', 'Z', 'a', 'b', 'd', 'e',
        'f', 'g', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 'sil',
        't', 'tS', 'u', 'w', 'x', 'z'
    ],
    'phonemes': [
        '/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/',
        '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/', '/sil/'
    ],
    'phones_to_phonemes': {
        'a': '/a/', 'e': '/e/', 'i': '/i/', 'o': '/o/', 'j': '/i/', 'w': '/u/', 'u': '/u/',
        'l': '/l/', 'r': '/R/', 'rr': '/r/', 't': '/t/', 'd': '/d/', 'D': '/d/', 'sil': '/sil/',
        '<p:>': '/sil/', 'm': '/m/', 'n': '/n/', 'N': '/n/', 'k': '/k/', 'g': '/g/', 'G': '/g/',
        'tS': '/tS/', 'T': '/tS/', 'f': '/f/', 'F': '/f/', 's': '/s/', 'S': '/s/', 'z': '/s/',
        'Z': '/s/', 'p': '/p/', 'b': '/b/', 'B': '/b/', 'L': '/L/', 'x': '/x/', 'jj': '/x/', 'J': '/x/'
    },
    'phonological_labels': {
        "vocalic": ["a", "e", "i", "o", "u", "w", "j"],
        "consonantal": ["b", "B", "d", "D", "f", "F", "k", "l", "m", "n", "N", "p", "r", "rr", "s", "Z", "T", "t", "g", "G", "tS", "S", "x", "jj", "J", "L", "z"],
        "back": ["a", "o", "u", "w"],
        "anterior": ["e", "i", "j"],
        "open": ["a", "e", "o"],
        "close": ["j", "i", "u", "w"],
        "nasal": ["m", "n", "N"],
        "stop": ["p", "b", "B", "t", "k", "g", "G", "tS", "d", "D"],
        "continuant": ["f", "F", "b", "B", "tS", "d", "D", "s", "Z", "T", "x", "jj", "J", "g", "G", "S", "L", "x", "jj", "J", "z"],
        "lateral": ["l"],
        "flap": ["r"],
        "trill": ["rr"],
        "voice": ["a", "e", "i", "o", "u", "w", "b", "B", "d", "D", "l", "m", "n", "N", "rr", "g", "G", "L", "j"],
        "strident": ["tS", "f", "F", "s", "Z", "T", "z",  "S"],
        "labial": ["m", "p", "b", "B", "f", "F"],
        "dental": ["t", "d", "D"],
        "velar": ["k", "g", "G"],
        "pause":  ["sil", "<p:>"]
    },
    'phonological_labels1': ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal'],
    'phonological_labels2': ['dental', 'consonantal', 'velar', 'flap', 'close', 'strident', 'continuant']
}


def compute_phones(
    phonet_obj: Phonet,
    audio_signal: np.ndarray,
    sample_rate: int,
    resampling_function: callable,
    PLLR: bool = False,
    target_sr: int = 128
) -> np.ndarray:
    """
    Compute phones from the audio file.

    Parameters
    ----------
    phonet_obj : Phonet
        An instance of the Phonet class for phoneme extraction.
    audio_signal: np.ndarray
        Audio signal array.
    sample_rate: int
        Sampling rate of the audio signal.
    target_sr : int
        Target sampling rate for the output phoneme probabilities.
    PLLR : bool
        Whether to return the PLLR (Phoneme Loglikelihood ratio). By default, True

    Returns
    -------
    np.ndarray
        If PLLR is True, returns the posterior probabilities of phonemes.
        If PLLR is False, returns the discrete phoneme sequence.
    """
    # Resample to 16 kHz as required by Phonet
    resampled_sample_rate = 16000
    audio_signal = resampling_function(
        array=audio_signal,
        original_sr=sample_rate,
        target_sr=resampled_sample_rate,
        axis=0
    )

    # This method extracts log-Mel-filterbank energies used as inputs of the model.
    # The output frequency is 100 Hz, which is the same as the time shift of the model.
    log_mel_filt_bank = phonet_obj.get_feat(
        audio_signal, resampled_sample_rate)

    # Calculate the number of frames represented in the audio signal
    number_of_frames = int(
        log_mel_filt_bank.shape[0]/phonet_obj.len_seq  # len_seq=40 always
    )

    # Segment the mels into sequences of len_seq frames
    input_features = []
    start, end = 0, phonet_obj.len_seq
    for j in range(number_of_frames):
        input_features.append(log_mel_filt_bank[start:end, :])
        start += phonet_obj.len_seq
        end += phonet_obj.len_seq

    # Standarize the input features
    input_features = np.stack(input_features, axis=0)
    input_features = input_features-phonet_obj.MU
    input_features = input_features/phonet_obj.STD

    # Get the predictions from the model and concatenate them to get a sequence
    probabilities = np.asarray(
        phonet_obj.model_phon.predict(input_features)
    )
    posterior_gram = np.concatenate(
        probabilities,
        axis=0
    )

    # time_shift is the time interval between frames
    total_audio_frames = int(
        len(audio_signal)/(phonet_obj.time_shift*resampled_sample_rate))
    posterior_gram = posterior_gram[:total_audio_frames]

    # posterior_prob: (num_frames, num_phones), original_fs ≈ 100 Hz
    num_target_frames = int(
        (audio_signal.shape[0] / resampled_sample_rate) * target_sr)
    posterior_gram = signal.resample(
        posterior_gram,
        num_target_frames,
        axis=0
    )
    if PLLR:
        return posterior_gram
    else:
        greedy_prediction = np.argmax(
            posterior_gram,
            axis=1
        )
        phone_sequence = [
            str(phonet_obj.phonemes[j])
            for j in greedy_prediction
        ]
        return phone_sequence


def extract_phonemes(
    audio_signal: np.ndarray,
    sample_rate: int,
    fixed_length: int,
    lexicon: dict,
    phonet_obj: Phonet,
    discrete: bool = False
) -> np.ndarray:
    posterior_prob = compute_phones(
        audio_signal=audio_signal,
        sample_rate=sample_rate,
        phonet_obj=phonet_obj,
        PLLR=True
    )
    posterior_prob = np.clip(
        posterior_prob, 1e-6, 1-1e-6
    )

    # Repeat last sample (probably silence)
    difference = len(posterior_prob) - fixed_length
    if difference > 0:
        posterior_prob = posterior_prob[:-difference]
    elif difference < 0:
        for i in range(np.abs(difference)):
            aux = posterior_prob[-1].copy()
            posterior_prob = np.vstack(
                (posterior_prob, aux.reshape(-1, 1).T))

    # Map phones to phonemes, making the sum
    posterior_prob_phonemes = np.zeros(
        shape=(posterior_prob.shape[0], len(lexicon['phonemes']))
    )
    for h, phone in enumerate(lexicon['phones']):
        phoneme_index = lexicon['phonemes'].index(
            lexicon['phones_to_phonemes'][phone]
        )
        posterior_prob_phonemes[:, phoneme_index] += posterior_prob[:, h]

    # Calculate posterior llr
    pllr = np.zeros(shape=posterior_prob_phonemes.shape)
    number_of_phonemes = posterior_prob_phonemes.shape[1]
    for ph in range(number_of_phonemes):
        if np.isinf(
                posterior_prob_phonemes[:, ph] /
                (1 - posterior_prob_phonemes[:, ph] + 1e-8)).any():
            print("There are infinite values in pllr")
        safe_vals = np.clip(
            posterior_prob_phonemes[:, ph] /
            (1 - posterior_prob_phonemes[:, ph] + 1e-8),
            1e-8, None)
        pllr[:, ph] = np.log10(
            safe_vals
        )

    # Centralize the pllr by subtracting the mean across phonemes for each frame
    pllr = np.nan_to_num(pllr, nan=0.0, posinf=0.0, neginf=0.0)
    pllr = pllr - np.mean(pllr, axis=1, keepdims=True)

    # Remove the silence phoneme
    pllr_without_silence = pllr[:, np.arange(
        number_of_phonemes) != lexicon['phonemes'].index('/sil/')]

    if discrete:
        pllr_without_silence = np.argmax(pllr_without_silence, axis=1)
    return pllr_without_silence
