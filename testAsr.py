from funasr import AutoModel

chunk_size = [0, 8, 4] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="/home/xfa/Documents/workspace/GPT-SoVITS/GPT-SoVITS/tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")

import soundfile
import os

wav_file = "uploaded_audio/kkk.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 600ms

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)[0].get('text')
    print(res+"\n")