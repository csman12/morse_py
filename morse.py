# morse.py: Encodes/Decodes morse code text & audio
import wave, struct, math, os, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, firwin, filtfilt

class Morse:
    # Encodes an alphanumeric string message into a morse code string based on symbol parameters 
    def encode(msg, sym_dot='.', sym_dash='-', sym_ltr_gap=' ', sym_word_gap='|'):
        if not isinstance(msg, str): raise Exception("Error, Invalid message type:", type(msg))
        if sym_dot == sym_dash: raise Exception("Error, dot symbol and dash symbol cannot be the same!")
        if sym_ltr_gap == sym_word_gap: raise Exception("Error, symbol letter gap and symbol word gap cannot be the same!")
        map_encode = Morse._get_encode_map(sym_dot=sym_dot, sym_dash=sym_dash)

        list_morse = [] 
        for i, ch in enumerate(msg):
            if ch.isspace():
                list_morse.pop() # remove prev sym_ltr_gap
                list_morse.append(' ' + sym_word_gap + ' ')
                continue
            if not ch.isalnum(): raise Exception("Error, character {} at index {} is not a valid alphanumeric character!".format(ch, i))
            list_morse.append(map_encode.get(ch))
            list_morse.append(sym_ltr_gap)
        list_morse.pop() # remove last sym_ltr_gap
        return "".join(list_morse)

    # Decodes a morse code string into an alphanumeric message using supplied symbol parameters
    def decode(morse_msg, sym_dot='.', sym_dash='-', sym_ltr_gap=' ', sym_word_gap='|'):
        if not isinstance(morse_msg, str): raise Exception("Error, Invalid morse message type:", type(msg))
        if sym_dot == sym_dash: raise Exception("Error, dot symbol and dash symbol cannot be the same!")
        if sym_ltr_gap == sym_word_gap: raise Exception("Error, symbol letter gap and symbol word gap cannot be the same!")
        map_decode = Morse._get_decode_map(sym_dot=sym_dot, sym_dash=sym_dash)

        list_chars = []
        for word in morse_msg.split(sym_word_gap):
            for morse in word.split(sym_ltr_gap):
                morse = morse.strip()
                if not morse: continue # skip if only white space was encountered
                char = map_decode.get(morse)
                if char is None:
                    raise Exception("Error, Unable to decode morse value:{}".format(morse))
                list_chars.append(char)
            list_chars.append(' ')

        return "".join(list_chars)
        
    # Create an audio wav file (saved as filename) of the morse code string supplied using the audio and symbol parameters supplied
    def create_morse_waveform(filename, morse_code, samp_rate=44100, samp_width=1, amp=1, wpm=16, freq=1000, sym_dot='.', sym_dash='-', sym_ltr_gap=' ', sym_word_gap='|'):
        if not morse_code: return None
        if freq > 1800 or freq < 100: raise Exception(f"Error, frequency must be between 100 and 1800 Hz! Value:{freq}")
        if samp_rate < 8000 or samp_rate > 44100: raise Exception(f"Error, sample rate must be between 8000 and 44100! Value:{samp_rate}")
        if wpm > 35 or wpm < 5: raise Exception(f"Error, words per minute must be between 5 and 35! Value{wpm}")
        if amp < 0.1 or amp > 1: raise Exception(f"Error, amplitude value must be between 0.1 and 1! Value:{amp}")
        if samp_width != 1 and samp_width != 2: raise Exception(f"Error, sample width must be 1 or 2 bytes! Value:{samp_width}")
        if sym_dot == sym_dash: raise Exception("Error, dot symbol and dash symbol cannot be the same!")
        if sym_ltr_gap == sym_word_gap: raise Exception("Error, symbol letter gap and symbol word gap cannot be the same!") 
        aud_file_path = Morse._validate_path_and_type(filename, ".wav") # validate and get abs file path

        # create audio waveforms for the dot & dash pulses used in constucting the audio waveform 
        num_channels = 1
        dot_length = 1.2/wpm # dot pulse length in seconds
        dash_length = dot_length * 3
        data_wave_dot = Morse._create_wave_seg(duration=dot_length, sample_rate=samp_rate, freq=freq, amp=amp, arc=True)
        data_wave_dash = Morse._create_wave_seg(duration=dash_length, sample_rate=samp_rate, freq=freq, amp=amp, arc=True)
        data_wave_sym_gap = Morse._create_wave_seg(duration=dot_length, sample_rate=samp_rate, freq=freq, amp=0, arc=False)
        
        aud_data = [] # used to hold the morse code audio data
        aud_data.extend(data_wave_sym_gap) # write an initial short off duration to the audio data
        morse_word_list = morse_code.split(sym_word_gap)
        last_morse_word_index = len(morse_word_list)-1
        for w_i, word in enumerate(morse_word_list): # add a corresponding audio segment to aud_data based on morse symbol read
            morse_char_list = word.split(sym_ltr_gap)
            last_morse_char_index = len(morse_char_list)-1
            for m_i, morse in enumerate(morse_char_list):
                morse = morse.strip()
                for sym in morse:
                    if sym == sym_dot: 
                        aud_data.extend(data_wave_dot)
                    elif sym == sym_dash:
                        aud_data.extend(data_wave_dash)
                    else: raise Exception("Error! Unknown morse symbol:{}".format(morse))
                    aud_data.extend(data_wave_sym_gap) # add space between dots and dashes
                if m_i != last_morse_char_index: # only write a letter gap if it is not the last char in the list
                    for i in range(2): aud_data.extend(data_wave_sym_gap) # write letter gap; ltr_gap duration = 3*dot_length
            if w_i != last_morse_word_index: # only write a word gap if it is not the last word in the list
                for i in range(6): aud_data.extend(data_wave_sym_gap) # write word gap; word_gap duration = 7*dot_length
         
        #aud_data = Morse._aud_filter(aud_data, samp_freq=sample_rate, lowcut=100, highcut=2000) # smooth any sharp audio edges
        Morse._write_aud_to_file(aud_data, aud_file_path, sample_rate=samp_rate, num_channels=num_channels, samp_width=samp_width)

    # Decodes an audio wave file (aud_filename) using a spectrogram (saved as spec_filename)
    # Returns decoded message, frequency channel used for decoding and the audio waveform details
    def decode_morse_waveform(aud_filename, spec_filename=None):
        sym_dot, sym_dash, sym_err, sym_gap, sym_ltr_gap, sym_word_gap = '.', '-', '#', '', ' ', '|'
        percent_carrier_threshold_for_signal_detection = 0.25 # multipled by peak carrier strengh to get carrier threshold
        percent_dot_dash_length_variation = 0.30
        percent_bin_deviation = 0.25 # deviation of pulse lengths used for sorting into bins

        # read in the audio data and perform a spectrogram on the audio wav
        aud_file_path = Morse._validate_path_and_type(aud_filename, ".wav") # validate and get abs file path
        aud_wav = Morse._get_aud_from_wav_file(aud_file_path)
        nperseg = math.floor(((200/44100)*aud_wav['params'].framerate)+160)
        f, t, Sxx = spectrogram(np.asarray(aud_wav['ch1']), fs=aud_wav['params'].framerate, return_onesided=True, scaling='spectrum', window=('hamming'), noverlap=100, nperseg=nperseg, nfft=512)

        # slice the spectrogram to only contain frequency between 0 and upper_freq
        upper_i, upper_freq = 0, 2000 # index and desired upper frequency limit for spectrogram
        for i in range(len(f)): # determine the index for freq right above freq limit
            if f[i] > upper_freq:
                upper_i = i
                upper_freq = f[i]
                break
        n_f = f[0:upper_i+1]
        n_Sxx = Sxx[0:upper_i+1]

        # write the spectrogram to file if filename was given
        if spec_filename is not None:
            spec_file_path = Morse._validate_path_and_type(spec_filename, ".png") # validate and get abs file path
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.pcolormesh(t, n_f, n_Sxx)
            plt.savefig(spec_file_path)
            plt.clf()

        # get the frequency peaks for all time bins of spectrogram
        f_peaks = [] # f_peaks is a list of peak freq index for each time bin
        for t_i in range(len(t)):
            f_peak = -1
            f_peak_i = 0
            for f_i in range(len(n_f)):
                if n_Sxx[f_i][t_i] > f_peak: 
                    f_peak = n_Sxx[f_i][t_i]
                    f_peak_i = f_i
            f_peaks.append(f_peak_i)

        # use hash table to count all of the frequency peaks
        f_peaks_cnt = {} # holds peak freq count in dict of form { freq index : [freq count, freq index]}
        for t_i in range(len(t)):
            peak_i =  f_peaks[t_i] #f_peaks.get(t_i)
            if f_peaks_cnt.get(peak_i) is None: f_peaks_cnt[peak_i] = [1, peak_i]
            else: f_peaks_cnt[peak_i][0] += 1

        # sort the peaks based on count and the pick the freq with the highest signal from the first two to use as our carrier freq
        peak_freqs = sorted(f_peaks_cnt.values(), key=lambda cnt: cnt[0], reverse=True)
        peak_freq1 = np.mean(n_Sxx[peak_freqs[0][1]])
        peak_freq2 = np.mean(n_Sxx[peak_freqs[1][1]])
        if peak_freq1 > peak_freq2: peak_freq_i = peak_freqs[0][1]
        else: peak_freq_i = peak_freqs[1][1]

        # create an on/off or true/false array of carrier frequency (peak_freq_i) for all time bins 
        carrier_peak_val = max(n_Sxx[peak_freq_i])
        signal_threshold = percent_carrier_threshold_for_signal_detection * carrier_peak_val
        on_off_time_array = []
        for val in n_Sxx[peak_freq_i]:
            if val > signal_threshold: on_off_time_array.append(True)
            else: on_off_time_array.append(False)

        # compact the on/off time array into a new array of on/off pulses
        on_cnt, off_cnt, last_val = 0, 0, None
        on_off_pulse_array = [] # each item in on_off_pulse_array has list form: [On/Off Type, pulse length(cnt), the index of this items bin(used in later step)]
        for val in on_off_time_array:
            if last_val is None:
                last_val = val
                if val == True: on_cnt = 1
                else: off_cnt = 1
            elif on_cnt > 0 and off_cnt > 0 and last_val != val: 
                if val == True:
                    on_off_pulse_array.append([True, on_cnt, None])
                    last_val=True
                    on_cnt=1
                else:
                    on_off_pulse_array.append([False, off_cnt, None])
                    last_val=False
                    off_cnt=1
            elif val is True:
                last_val = True
                on_cnt += 1
            elif val is False:
                last_val = False
                off_cnt += 1
            else: raise Exception("Pulse Decoding Error!")
        if last_val == True: # write the trailing two pulse counts
            if (off_cnt > 0): on_off_pulse_array.append([False, off_cnt, None])
            on_off_pulse_array.append([True, on_cnt, None])
        else: 
            if (on_cnt > 0): on_off_pulse_array.append([True, on_cnt, None])
            on_off_pulse_array.append([False, off_cnt, None])

        # put the on/off pulses into bins that are of similar size; a 'similar' bin is based on variable percent_bin_deviation
        bins = [] # each bin (list item) has a head list at index 0 as a list of [bin average, bin count, bin morse symbol]
        for item in on_off_pulse_array:
            for i, bin in enumerate(bins):
                diff = abs(bin[0][0]-item[1]) # difference = absolute(bin_average_pulse_length - current_items_pulse_length)
                deviation = bin[0][0] * percent_bin_deviation # allowed deviation = bin_average * percent_bin_deviation
                if diff <= deviation and bin[1][0]==item[0]: # check if pulse item should go into this bin (within deviation and same on/off type)
                    item[2] = i # set the items bin index of the bin it is going into
                    bin.append(item)
                    bin[0][0] *= bin[0][1] # bin_average = bin_average * bin_count; => getting the bin sum back
                    bin[0][1] += 1 # bin_count = bin_count + 1
                    bin[0][0] = (bin[0][0] + item[1]) / bin[0][1] # bin_average = (bin_sum + item_pulse_length) / bin_count
                    break
            else: # if we iterate thru all bins without finding a place for current item, then create a new bin
                item[2] = len(bins) # set the bin index for the current item
                bins.append([[item[1], 1, None], item]) # add the bin header followed by the pulse item

        # create a list of true bins and determine what morse symbol each bin should be
        true_bins = []
        for i, bin in enumerate(bins):
            if bin[1][0] == True:
                true_bins.append(bin)
        if len(true_bins) < 2: raise Exception("Error, Unable to decode single pulse!")
        else:
            true_bins.sort(key=lambda bin: bin[0][1], reverse=True) # sort by bin item count in decreasing order
            if true_bins[0][0][0] > true_bins[1][0][0]: # if first_bin_pulse_average > second_bin_pulse_average
                dash_bin, dot_bin = true_bins[0], true_bins[1]
            else:
                dash_bin, dot_bin = true_bins[1], true_bins[0]
            dash_bin[0][2] = sym_dash
            dot_bin[0][2] = sym_dot    
            if (dash_bin[0][0]-(3*dot_bin[0][0])) > (dash_bin[0][0]*percent_dot_dash_length_variation): # check that dash pulse length is close to dot length times 3
                raise Exception("Error, Unable to determine dot & dash pulse lengths")
            for i in range(2, len(true_bins)):
                true_bins[i][0][2] = sym_err # put an error symbol for all remaining true bin pulses

        # create a list of false bins and determine what each bin should be
        dot_pulse_length = math.floor(dot_bin[0][0])
        exp_gap = [dot_pulse_length, dot_pulse_length*3, dot_pulse_length*7] # expected off pulse lengths = [ sym_gap, ltr_gap, word_gap ]
        false_bins = []
        for i, bin in enumerate(bins):
            if bin[1][0] == False:
                false_bins.append(bin)
        # calc the difference between bin and each expected gap length; the bin belongs to the morse symbol gap with the least difference
        for bin in false_bins:
            min_val = 100000
            min_i = 3
            for i in range(0, 3):
                diff = abs(bin[0][0]-exp_gap[i])
                if diff < min_val:
                    min_val = diff
                    min_i = i
            if min_i == 0: bin[0][2] = sym_gap
            elif min_i == 1: bin[0][2] = sym_ltr_gap
            else: bin[0][2] = sym_word_gap

        # create a morse code symbol array based on the bin the pulse item belongs too
        morse = []
        for pulse in on_off_pulse_array:
            morse.append(bins[pulse[2]][0][2])

        #wpm = 1.2 / ( dot_pulse_length * ((aud_wav['params'].nframes / aud_wav['params'].framerate) / len(t)) )
        freq_ch = n_f[peak_freq_i]
        
        print("Decoding:", "".join(morse))
        dec_msg = Morse.decode("".join(morse), sym_dot=sym_dot, sym_dash=sym_dash, sym_ltr_gap=sym_ltr_gap, sym_word_gap=sym_word_gap)
        return dec_msg, freq_ch, copy.copy(aud_wav['params'])

    # throws exception if file_path is not valid; returns absolute path
    def _validate_path_and_type(file_path=None, allowed_ext=None):
        if file_path is None or file_path=="": raise Exception("Error, no file name was given!")
        aud_dir, aud_file = os.path.split(file_path)
        if not os.path.isabs(aud_dir): aud_dir = os.path.abspath(aud_dir) 
        if aud_dir and not os.path.isdir(aud_dir): raise Exception(f"Error, invalid directory for file: {aud_dir}")
        if aud_dir[0] == "\\" and platform.system() == "Windows": raise Exception(f"Error, invalid absolute path for file: {aud_dir}")
        aud_filename, aud_ext = os.path.splitext(aud_file)
        if allowed_ext and aud_ext not in allowed_ext: raise Exception(f"Error, invalid file extension: {aud_ext}")
        if not aud_filename: raise Exception(f"Error, invalid file name: {aud_filename}")
        return os.path.join(aud_dir, aud_file) # re-join with abs dir path

    def _create_wave_seg(duration=0, sample_rate=44100, freq=1000, amp=1, arc=False):
        delta_t = 1/sample_rate
        omega = 2*np.pi*freq
        num_samples = duration//delta_t
        time = np.arange(num_samples)*delta_t
        if not amp: wav = [0]*len(time)
        else: wav = np.sin(omega*time) * amp
        if not arc:
            return wav
        # add an enter and exit arc ramp to the audio wave
        arc_freq = 1/(duration*2)
        arc_omega = 2*np.pi*arc_freq
        arc_env = np.sin(arc_omega*time)*2
        for i in range(len(wav)):
            if arc_env[i] > 1: continue
            else: wav[i] = wav[i] * arc_env[i]
        return wav

    def _aud_filter(data, samp_freq, lowcut, highcut, numtaps=500):
        low = lowcut / (0.5 * samp_freq)  # is in half-cycles / sample
        high = highcut / (0.5 * samp_freq)

        taps = firwin(numtaps, [low,high], window='hamming', pass_zero=False)
        y = filtfilt(taps, 1.0, data)

        for i in range(len(y)): 
            y[i] = min(y[i], 1) if y[i]>=0 else max(y[i], -1)
        return y

    def _write_aud_to_file(audio, filename, sample_rate=44100, num_channels=1, samp_width=1):
        for val in audio:
            if val > 1 or val < -1: raise Exception(f"Bad audio value:{val}. Must be -1 <= value <= 1")
        # create waveform sample data based on samp_width
        if samp_width == 2: 
            int_size = 32767 # 16-bit sample size
            samp_data = [(int)(val*int_size) for val in audio]
        elif samp_width == 1: 
            int_size = 127 # 8-bit sample size
            samp_data = [(int)((val+1)*int_size) for val in audio] # add offset to val of 1 to make all values positive for writing bytes
        else: raise Exception(f"Invalid sample width size:{samp_width}")
        
        # if using dual channels, make a copy of the wave sample data for both channels
        if num_channels == 2:
            aud_data = [0]*(len(samp_data)*2)
            for i in range(0, len(aud_data), 2):
                aud_data[i] = samp_data[i//2] # left channel audio
                aud_data[i+1] = samp_data[i//2] # right channel audio
        elif num_channels == 1:
            aud_data = samp_data # mono channel
        else: raise Exception(f"Invalid number of channels:{num_channels}")

        # open the wave file and write the audio data to it based on samp_width
        obj = wave.open(filename,'wb')
        obj.setnchannels(num_channels)
        obj.setsampwidth(samp_width)
        obj.setframerate(sample_rate)
        if samp_width == 1: obj.writeframesraw(struct.pack('<{}B'.format(len(aud_data)), *aud_data))
        elif samp_width == 2: obj.writeframesraw(struct.pack('<{}h'.format(len(aud_data)), *aud_data))
        obj.close() 

    def _get_aud_from_wav_file(filename, rtn_raw=False):
        obj = wave.open(filename,'rb')
        params = obj.getparams()
        ch1, ch2 = [], []
        chunk_size = 4096

        for size in range(0, params.nframes, chunk_size):
            bytes_read = obj.readframes(chunk_size)
            if params.nchannels == 1 and params.sampwidth == 1:
                ch1.extend(bytes_read)
            elif params.nchannels == 2 and params.sampwidth == 1:
                ch1.extend([bytes_read[i] for i in range(0, len(bytes_read), 2)]) # left channel
                ch2.extend([bytes_read[i] for i in range(1, len(bytes_read), 2)]) # right channel
            elif params.nchannels == 1 and params.sampwidth == 2:
                ch1.extend(struct.unpack('<{}h'.format(len(bytes_read)//2), bytes_read))
            elif params.nchannels == 2 and params.sampwidth == 2:
                chan_bytes = bytearray(b'\x00') * (len(bytes_read)//2)
                for i in range(0, len(bytes_read)//2, 2):
                    chan_bytes[i] = bytes_read[i*2]
                    chan_bytes[i+1] = bytes_read[(i*2)+1]
                ch1.extend(struct.unpack('<{}h'.format(len(bytes_read)//4), chan_bytes)) # left channel
                for i in range(2, len(bytes_read)//2, 2):
                    chan_bytes[i] = bytes_read[i*2]
                    chan_bytes[i+1] = bytes_read[(i*2)+1]
                ch2.extend(struct.unpack('<{}h'.format(len(bytes_read)//4), chan_bytes)) # right channel
            else: raise Exception("Invalid channel and sample width size!")
        obj.close()

        rtn_dict = { 'params':params, 'ch1':ch1, 'ch2':ch2, 'rtn_raw':rtn_raw }
        if rtn_raw:
            return rtn_dict
        
        int_size = (2**(8*params.sampwidth)//2)-1
        for i in range(len(ch1)): ch1[i] = ch1[i]/int_size
        for i in range(len(ch2)): ch2[i] = ch2[i]/int_size 
        return rtn_dict

    def _get_encode_map(sym_dot='.', sym_dash='-'):
        if not sym_dot or not sym_dash: return Morse.map_default_encode.copy()
        if sym_dot=='.' and sym_dash=='-': return Morse.map_default_encode.copy()

        new_encode_map = {}
        for key, value in Morse.map_default_encode.items():
            new_encode_map[key] = value.replace('.', sym_dot).replace('-', sym_dash)
        return new_encode_map

    def _get_decode_map(sym_dot='.', sym_dash='-'):
        if not sym_dot or not sym_dash: return Morse.map_default_decode.copy()
        if sym_dot=='.' and sym_dash=='-': return Morse.map_default_decode.copy()

        new_decode_map = {}
        for key, value in Morse.map_default_decode.items():
            new_decode_map[key.replace('.', sym_dot).replace('-', sym_dash)] = value
        return new_decode_map

    map_default_encode = {
        "A": ".-", "a": ".-",
        "B": "-...", "b": "-...",
        "C": "-.-.", "c": "-.-.",
        "D": "-..", "d": "-..",
        "E": ".", "e": ".",
        "F": "..-.", "f": "..-.",
        "G": "--.", "g": "--.",
        "H": "....", "h": "....",
        "I": "..", "i": "..",
        "J": ".---", "j": ".---",
        "K": "-.-", "k": "-.-",
        "L": ".-..", "l": ".-..",
        "M": "--", "m": "--",
        "N": "-.", "n": "-.",
        "O": "---", "o": "---",
        "P": ".--.", "p": ".--.",
        "Q": "--.-", "q": "--.-",
        "R": ".-.", "r": ".-.",
        "S": "...", "s": "...",
        "T": "-", "t": "-",
        "U": "..-", "u": "..-",
        "V": "...-", "v": "...-",
        "W": ".--", "w": ".--",
        "X": "-..-", "x": "-..-",
        "Y": "-.--", "y": "-.--",
        "Z": "--..", "z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----"
    }

    map_default_decode = {
        ".-": "a",
        "-...": "b",
        "-.-.": "c",
        "-..": "d",
        ".": "e",
        "..-.": "f",
        "--.": "g",
        "....": "h",
        "..": "i",
        ".---": "j",
        "-.-": "k",
        ".-..": "l",
        "--": "m",
        "-.": "n",
        "---": "o",
        ".--.": "p",
        "--.-": "q",
        ".-.": "r",
        "...": "s",
        "-": "t",
        "..-": "u",
        "...-": "v",
        ".--": "w",
        "-..-": "x",
        "-.--": "y",
        "--..": "z",
        ".----": "1",
        "..---": "2",
        "...--": "3",
        "....-": "4",
        ".....": "5",
        "-....": "6",
        "--...": "7",
        "---..": "8",
        "----.": "9",
        "-----": "0"
    }

if __name__ == "__main__":
    test_msg = "this is a test 1 2 3 4 5 6 7 8 9 0"
    print("Testing Message:", test_msg)
    morse = Morse.encode(test_msg)
    print("Morse Code:", morse)
    dec_txt_msg = Morse.decode(morse)
    print("Decoded Morse Message:", dec_txt_msg)
    Morse.create_morse_waveform("sound.wav", morse, samp_rate=44100, wpm=16, freq=650)
    print("Created audio file of morse code!")
    dec_aud_msg = Morse.decode_morse_waveform("sound.wav", "test.png")
    print("Decoded Audio Message:", dec_aud_msg)
